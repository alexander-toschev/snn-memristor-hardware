# collect_counts_plus — «probe» скрытого слоя SNN, а не финальный test.
#
# Зачем:
# - Быстрая диагностика: проверяем, есть ли полезные признаки в текущем спайковом коде,
#   не трогая веса (STDP заморожен).
# - Контроль гиперпараметров: видно влияние rate_scale, порога, ингибиции, tau на
#   информативность признаков до тяжёлых end-to-end замеров.
# - Ранний детект регрессий: если после рефакторинга слой «оглох», это сразу видно.
#
# Что делает:
# 1) На каждом образце гоняет сеть вперёд T тактов (без обучения).
# 2) Снимает:
#    - counts: суммарные спайки по каждому нейрону скрытого слоя (N признаков),
#    - WTA-hist: 10-мерную гистограмму победителей по классам (через label_map).
# 3) Склеивает [counts || WTA-hist] → строка признаков, возвращает X, y
#    (и опционально debug-инфо).
#
# Как трактовать:
# - mean sum per sample (по скрытому слою) ≈ 5–20 — здоровая плотность для простого рид-аута.
#   Если ≈ 0–1, слой «тихий» → рид-аут будет близок к случайности.
# - winners_unique ~ n_hidden и низкий HHI (≈ 0.02–0.10) — роли нейронов распределены.
# - Быстрый рид-аут (Linear/MLP поверх X) должен быть заметно > 0.10 (случайность на MNIST).
#
# Временные «подсветки» только на время окна (всё откатывается):
# - vt_eval_offset / threshold_abs — временно понижает порог, чтобы увидеть активность.
# - encoder_rate_boost — умножает интенсивность Пуассона (больше входных спайков).
# - temp_disable_inh — временно снимает латеральное торможение (LIF→LIF).
#
# Когда использовать:
# - Во время тюнинга и A/B сравнения конфигураций — да.
# - Как smoke-test (например, на 1000 сэмплов) — да.
# - В финальном прод-пайплайне — опционально.
#
# Пример:
#     Xtr, ytr = collect_counts_plus(
#         net, lif_layer, enc, ds_train, 60000, T=cfg.time, label_map=label_map_build,
#         # опции подсветки на сборе (по одному за раз):
#         # encoder_rate_boost=3.0,
#         # threshold_abs=0.01,
#         # vt_eval_offset=-0.25,
#         # temp_disable_inh=True,
#         progress=True, desc="Collect train"
#     )
#
# Возвращает:
# - (X, y) или (X, y, debug_info), где X.shape = [n_samples, n_hidden + 10].


# counts_readout.py
from __future__ import annotations
from typing import Tuple, Optional

import torch
from torch import Tensor
from torchvision import transforms
from bindsnet.datasets import MNIST
from bindsnet.network.monitors import Monitor
from tqdm import tqdm

__all__ = ["collect_counts_plus", "make_mnist_datasets", "zscore_normalize"]



@torch.no_grad()
def collect_counts_plus_cuda(
    net,
    lif_layer,
    encoder,
    ds,
    n_samples: int,
    T: int,
    label_map,
    vt_eval_offset: float = 0.0,
    debug_every: int = 1000,
    verify_offset: bool = True,
    return_debug: bool = False,
    progress: bool = True,
    desc: str | None = None,
    ncols: int = 100,
    leave: bool = False,
    device: torch.device | str | None = None,
    output_device: torch.device | str = "cpu",  # "cuda" если хочешь вернуть на GPU
):
    """
    CUDA-версия collect_counts_plus:
    - net / lif_layer / label_map / вычисления -> на GPU
    - X/y можно вернуть на CPU (по умолчанию), чтобы дальше sklearn и т.п.

    Возвращает: (X, y) или (X, y, debug_info)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    output_device = torch.device(output_device)

    # 0) замораживаем STDP + переносим параметры update_rule на нужный девайс (если есть)
    for c in net.connections.values():
        if hasattr(c, "update_rule"):
            # nu в BindsNET часто ожидается tuple тензоров
            c.update_rule.nu = (torch.as_tensor(0.0, device=device), torch.as_tensor(0.0, device=device))

    # 1) переносим сеть/слои на GPU (если доступно)
    # В разных версиях BindsNET бывает по-разному, поэтому делаем "надёжно":
    if hasattr(net, "to"):
        net.to(device)
    if hasattr(lif_layer, "to"):
        lif_layer.to(device)

    for l in getattr(net, "layers", {}).values():
        if hasattr(l, "to"):
            l.to(device)
    for c in getattr(net, "connections", {}).values():
        if hasattr(c, "to"):
            c.to(device)

    # 2) монитор скрытого слоя (будет хранить state_vars там же, где layer)
    mon = Monitor(lif_layer, state_vars=("s",), time=T)
    net.add_monitor(mon, name="lif_tmp_counts_plus")

    N = lif_layer.n

    # Фичи/лейблы собираем на GPU, чтобы не гонять тензоры туда-сюда каждый шаг
    X = torch.empty((n_samples, N + 10), device=device, dtype=torch.float32)
    y = torch.empty((n_samples,), device=device, dtype=torch.long)

    # label_map на GPU
    lm = torch.as_tensor(label_map, dtype=torch.long, device=device)

    # 3) доступ к порогам
    vt_attr = "v_thresh" if hasattr(lif_layer, "v_thresh") else "thresh"
    vt0 = getattr(lif_layer, vt_attr)

    if torch.is_tensor(vt0):
        vt_backup = vt0.detach().clone()
    else:
        vt_backup = torch.as_tensor(float(vt0), dtype=torch.float32, device=device)

    # --- счётчики/диагностика оффсета (без частых .item()) ---
    applied_cnt = 0
    restored_ok = 0

    # аккумулируем дельту на GPU, потом один раз посчитаем среднее
    delta_sum = torch.zeros((), device=device, dtype=torch.float32)
    delta_n = 0

    vt_before_samples, vt_after_apply_samples, vt_after_restore_samples = [], [], []

    def _apply_offset(offset: float):
        nonlocal applied_cnt, delta_sum, delta_n
        cur = getattr(lif_layer, vt_attr)

        if torch.is_tensor(cur):
            vt_before = cur.detach().clone()
            new_vt = cur.detach().clone()
            new_vt.add_(offset).clamp_(-10.0, 10.0)
            setattr(lif_layer, vt_attr, new_vt)
            applied_cnt += 1

            # Δmean(vt) считаем на GPU (без .item() каждый раз)
            delta_sum = delta_sum + (new_vt.float().mean() - vt_before.float().mean())
            delta_n += 1
            return vt_before, new_vt
        else:
            # скалярный thresh: тоже ведём на device
            vt_before = torch.as_tensor(float(cur), device=device)
            new_vt = torch.as_tensor(max(0.0, min(10.0, float(cur) + float(offset))), device=device)
            setattr(lif_layer, vt_attr, float(new_vt.item()))  # тут .item() разово (иначе setattr не поймёт tensor)
            applied_cnt += 1

            delta_sum = delta_sum + (new_vt - vt_before)
            delta_n += 1
            return vt_before, new_vt

    def _restore_vt():
        cur = getattr(lif_layer, vt_attr)
        if torch.is_tensor(cur):
            cur.copy_(vt_backup)
        else:
            setattr(lif_layer, vt_attr, float(vt_backup.item()))

    # 4) итератор
    iterator = range(n_samples)
    if progress:
        iterator = tqdm(
            iterator,
            desc=desc or f"Collect counts+WTA (CUDA) T={T}",
            ncols=ncols,
            leave=leave,
            mininterval=2.0,
            miniters=200,
        )

    for i in iterator:
        sample = ds[i]
        x_i, y_i = sample["image"], int(sample["label"])

        # A) оффсет порога на время окна
        if vt_eval_offset != 0.0:
            vt_before, vt_after = _apply_offset(vt_eval_offset)

            # редкие сэмплы для проверки (тут .item() допустим редко)
            if (i < 5) or (debug_every and ((i + 1) % debug_every == 0)):
                vt_before_samples.append(float(vt_before.float().mean().detach().cpu()))
                vt_after_apply_samples.append(float(vt_after.float().mean().detach().cpu()))

        # B) прогон (вход + encoder на GPU)
        # переносим image на GPU (если это torch.Tensor)
        if torch.is_tensor(x_i):
            x_i = x_i.to(device, non_blocking=True)
        spikes_in = encoder(x_i)

        # на случай, если encoder вернул CPU
        if torch.is_tensor(spikes_in) and spikes_in.device != device:
            spikes_in = spikes_in.to(device, non_blocking=True)

        net.run(inputs={"Input": spikes_in}, time=T)

        # C) чтение спайков
        s = mon.get("s")  # ожидаем [T, batch, N] или [T, N] в зависимости от версии/конфига
        # приводим к [T, N]
        if s.dim() == 3:
            s = s[:, 0, :]
        s = s.to(dtype=torch.float32, device=device)

        counts = s.sum(0)  # [N]

        # D) WTA-гистограмма
        active_t = (s.sum(1) > 0)
        if active_t.any():
            winners_t = s[active_t].argmax(dim=1)      # [K]
            cls_idx = lm[winners_t]                    # [K]
            cls_idx = cls_idx[cls_idx >= 0]
            if cls_idx.numel() > 0:
                hist = torch.bincount(cls_idx, minlength=10).to(torch.float32)
                hist = hist / active_t.sum().to(torch.float32)
            else:
                hist = torch.zeros(10, dtype=torch.float32, device=device)
        else:
            hist = torch.zeros(10, dtype=torch.float32, device=device)

        X[i] = torch.cat([counts, hist], dim=0)
        y[i] = y_i

        # E) сброс состояний + восстановление порога
        net.reset_state_variables()
        mon.reset_state_variables()

        if vt_eval_offset != 0.0:
            _restore_vt()
            cur = getattr(lif_layer, vt_attr)

            # выборочная проверка восстановления
            if (i < 5) or (debug_every and ((i + 1) % debug_every == 0)):
                cur_t = cur if torch.is_tensor(cur) else torch.as_tensor(float(cur), device=device)
                vt_after_restore_samples.append(float(cur_t.float().mean().detach().cpu()))

            if torch.is_tensor(cur):
                restored_ok += int(torch.allclose(cur, vt_backup, atol=1e-6))
            else:
                restored_ok += int(abs(float(cur) - float(vt_backup.item())) < 1e-6)

        # F) постфикс (редко, чтобы не убивать IOPub и не синхронизировать GPU постоянно)
        if progress and (debug_every and ((i + 1) % debug_every == 0)):
            # тут .detach().cpu() намеренно редко
            sum_out = float(counts.sum().detach().cpu())
            iterator.set_postfix_str(f"sum_out={sum_out:.2f} vt_off={vt_eval_offset:+.3f}")

        if (not progress) and (debug_every and ((i + 1) % debug_every == 0)):
            sum_out = float(counts.sum().detach().cpu())
            print(f"[collect {i+1}/{n_samples}] sum_out={sum_out:.2f} vt_off={vt_eval_offset:+.3f}")

    # 5) снятие монитора
    net.monitors.pop("lif_tmp_counts_plus", None)

    # 6) финальная сводка по оффсету
    avg_delta = float((delta_sum / max(1, delta_n)).detach().cpu()) if delta_n > 0 else 0.0
    debug_info = {
        "requested_offset": float(vt_eval_offset),
        "applied_count": int(applied_cnt),
        "avg_delta_mean_vt": avg_delta,
        "restore_ok_fraction": (float(restored_ok / max(1, applied_cnt)) if applied_cnt > 0 else 1.0),
        "vt_before_samples": vt_before_samples,
        "vt_after_apply_samples": vt_after_apply_samples,
        "vt_after_restore_samples": vt_after_restore_samples,
        "device": str(device),
    }

    if verify_offset:
        print(
            "[collect_counts_plus_cuda] offset summary:",
            f"requested={vt_eval_offset:+.3f}, applied={applied_cnt}/{n_samples}, "
            f"avgΔvt={avg_delta:.3f}, restore_ok={restored_ok}/{max(1, applied_cnt)}"
        )

    # 7) вывод на нужный девайс
    if output_device.type == "cpu":
        X_out = X.detach().cpu()
        y_out = y.detach().cpu()
    else:
        X_out = X
        y_out = y

    return (X_out, y_out, debug_info) if return_debug else (X_out, y_out)



def make_mnist_datasets(transform: Optional[transforms.Compose] = None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    ds_train = MNIST(root="./data", train=True, download=True, transform=transform)
    ds_test  = MNIST(root="./data", train=False, download=True, transform=transform)
    return ds_train, ds_test

@torch.no_grad()
def zscore_normalize(Xtr: Tensor, Xte: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    mu = Xtr.mean(0, keepdim=True)
    sigma = Xtr.std(0, keepdim=True).clamp_min(eps)
    return (Xtr - mu) / sigma, (Xte - mu) / sigma, mu, sigma


import torch
import numpy as np
from torch.utils.data import DataLoader



@torch.no_grad()
def collect_counts_plus_fast(
    net,
    lif_layer,
    encoder,
    ds,
    n_samples: int,
    T: int,
    label_map,
    *,
    spikes_transform=None,
    on_batch=None,
    vt_eval_offset: float = 0.0,
    debug_every: int = 1000,
    verify_offset: bool = True,
    return_debug: bool = False,
    progress: bool = True,
    desc: str | None = None,
    ncols: int = 100,
    leave: bool = False,
    device: torch.device | str | None = None,
    output_device: torch.device | str = "cpu",
    move_net: bool = True,          # не трогаем net.to(...) если не нужно
    batch_size: int = 64,            # батчинг
    num_workers: int = 0,            # dataloader
    pin_memory: bool = True,         # ускоряет H2D
    # --- НОВОЕ: «подсветка» ТОЛЬКО на время сбора, всё откатывается ---
    encoder_rate_boost: float = 1.0,       # множитель пуассона на сборе
    threshold_abs: float | None = None,    # абсолютный порог на окно, напр. 0.01
    temp_disable_inh: bool = False,        # временно занулять LIF→LIF ингибицию
):
    """
    Сбор признаков: [counts per neuron, 10D WTA-class histogram].
    GPU-оптимизировано: батчи, минимум синхронизаций, без лишних переносов.
    Временные «подсветки» (encoder_rate_boost, threshold_abs, temp_disable_inh)
    действуют только в окне и откатываются.
    Возвращает: (X, y) или (X, y, debug_info).
    """
    import math
    from torch.utils.data import DataLoader
    from bindsnet.network.monitors import Monitor
    from tqdm import tqdm

    # --- helpers ---
    def _poisson_boost_batch(x_b, T, base_scale, boost, dev):
        # x_b: [B,1,28,28] -> spikes [T,B,784]
        B = x_b.shape[0]
        x_flat = x_b.view(B, -1).clamp(0, 1)
        lam = x_flat * (base_scale * boost)
        p = 1.0 - torch.exp(-lam)
        rand = torch.rand((T, B, lam.shape[1]), device=dev)
        return (rand < p).float()  # [T,B,784]

    # 0) девайс
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    output_device = torch.device(output_device)

    # 0.1) freeze STDP
    for c in net.connections.values():
        if hasattr(c, "update_rule"):
            c.update_rule.nu = (torch.as_tensor(0.0, device=device),
                                torch.as_tensor(0.0, device=device))

    # 1) перенос сети/слоёв по требованию
    if move_net:
        if hasattr(net, "to"): net.to(device)
        if hasattr(lif_layer, "to"): lif_layer.to(device)
        for l in getattr(net, "layers", {}).values():
            if hasattr(l, "to"): l.to(device)
        for c in getattr(net, "connections", {}).values():
            if hasattr(c, "to"): c.to(device)

    # 1.1) найдём латеральку для временного выключения (если понадобится)
    rec_conn = None
    if temp_disable_inh:
        for (src, dst), conn in net.connections.items():
            if src == "LIF" and dst == "LIF":
                rec_conn = conn
                break
    Wrec_backup = rec_conn.w.clone() if rec_conn is not None else None

    # 2) монитор
    mon = Monitor(lif_layer, state_vars=("s",), time=T)
    net.add_monitor(mon, name="lif_tmp_counts_plus_fast")

    N = lif_layer.n
    # For large conv hidden layers, X can be huge (n_samples*(N+10)).
    # Allow streaming to a disk-backed memmap to avoid OOM.
    use_memmap = False
    mm_path = None
    try:
        use_memmap = bool(getattr(net, "_use_memmap", False))
    except Exception:
        use_memmap = False

    if use_memmap:
        import numpy as np
        mm_path = getattr(net, "_memmap_path", None)
        if mm_path is None:
            raise ValueError("use_memmap requested but net._memmap_path is None")
        X = np.memmap(str(mm_path), mode="w+", dtype=np.float32, shape=(n_samples, N + 10))
        y_path = str(mm_path) + ".y.memmap"
        y = np.memmap(y_path, mode="w+", dtype=np.int64, shape=(n_samples,))
    else:
        X = torch.empty((n_samples, N + 10), device=output_device, dtype=torch.float32)
        y = torch.empty((n_samples,), device=output_device, dtype=torch.long)

    # label_map и пороги
    lm = torch.as_tensor(label_map, dtype=torch.long, device=device)
    vt_attr = "v_thresh" if hasattr(lif_layer, "v_thresh") else "thresh"
    vt0 = getattr(lif_layer, vt_attr)
    vt_backup = vt0.detach().clone() if torch.is_tensor(vt0) else torch.as_tensor(float(vt0), dtype=torch.float32, device=device)

    applied_cnt = 0
    restored_ok = 0
    delta_sum = torch.zeros((), device=device, dtype=torch.float32)
    delta_n = 0
    vt_before_samples, vt_after_apply_samples, vt_after_restore_samples = [], [], []

    def _apply_offset(offset: float):
        nonlocal applied_cnt, delta_sum, delta_n
        cur = getattr(lif_layer, vt_attr)
        if torch.is_tensor(cur):
            vt_before = cur.detach().clone()
            new_vt = cur.detach().clone()
            new_vt.add_(offset).clamp_(-10.0, 10.0)
            setattr(lif_layer, vt_attr, new_vt)
            applied_cnt += 1
            delta_sum.add_(new_vt.float().mean() - vt_before.float().mean())
            delta_n += 1
            return vt_before, new_vt
        else:
            vt_before = torch.as_tensor(float(cur), device=device)
            new_vt = torch.as_tensor(max(0.0, min(10.0, float(cur) + float(offset))), device=device)
            setattr(lif_layer, vt_attr, float(new_vt.item()))
            applied_cnt += 1
            delta_sum.add_(new_vt - vt_before)
            delta_n += 1
            return vt_before, new_vt

    def _apply_abs(value: float):
        nonlocal applied_cnt
        cur = getattr(lif_layer, vt_attr)
        if torch.is_tensor(cur):
            vt_before = cur.detach().clone()
            new_vt = torch.full_like(cur, float(value))
            setattr(lif_layer, vt_attr, new_vt)
            applied_cnt += 1
            return vt_before, new_vt
        else:
            vt_before = torch.as_tensor(float(cur), device=device)
            setattr(lif_layer, vt_attr, float(value))
            applied_cnt += 1
            return vt_before, torch.as_tensor(float(value), device=device)

    def _restore_vt():
        cur = getattr(lif_layer, vt_attr)
        if torch.is_tensor(cur):
            cur.copy_(vt_backup)
        else:
            setattr(lif_layer, vt_attr, float(vt_backup.item()))

    # 4) DataLoader на первые n_samples
    class _TakeN(torch.utils.data.Dataset):
        def __init__(self, base, n): self.base, self.n = base, n
        def __len__(self): return self.n
        def __getitem__(self, i): return self.base[i]

    def _collate(batch):
        xs = torch.stack([b["image"] for b in batch], dim=0)  # [B,1,28,28]
        ys = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        return xs, ys

    dl = DataLoader(
        _TakeN(ds, n_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        collate_fn=_collate,
        drop_last=False,
    )

    # tqdm can raise BrokenPipeError when stdout is not a real TTY (background runs).
    try:
        iterator = tqdm(
            dl,
            desc=desc or f"Collect counts+WTA (T={T}, bs={batch_size})",
            ncols=ncols,
            leave=leave,
            mininterval=2.0,
            miniters=1,
            disable=not progress,
        )
    except (BrokenPipeError, OSError):
        iterator = dl

    write_pos = 0

    for step, (x_b, y_b) in enumerate(iterator):
        B = x_b.shape[0]
        if on_batch is not None:
            try:
                on_batch(step=step, write_pos=write_pos, batch_size=B, n_samples=n_samples)
            except Exception:
                pass

        # A) ингибицию глушим на окно (если нужно)
        if rec_conn is not None:
            with torch.no_grad():
                rec_conn.w.zero_()

        # B) порог на окно: threshold_abs приоритетнее offset
        vt_before = vt_after = None
        if threshold_abs is not None:
            vt_before, vt_after = _apply_abs(threshold_abs)
        elif vt_eval_offset != 0.0:
            vt_before, vt_after = _apply_offset(vt_eval_offset)

        if (vt_before is not None) and ((write_pos < 5) or (debug_every and (((write_pos + B) // debug_every) != (write_pos // debug_every)))):
            vt_before_samples.append(float(vt_before.float().mean().detach().cpu()))
            vt_after_apply_samples.append(float(vt_after.float().mean().detach().cpu()))

        # C) вход
        x_b = x_b.to(device, non_blocking=True)
        if (encoder_rate_boost != 1.0) and hasattr(encoder, "rate_scale"):
            spikes_in = _poisson_boost_batch(x_b, T, getattr(encoder, "rate_scale", 1.0), encoder_rate_boost, device)  # [T,B,784]
            if spikes_transform is not None:
                spikes_in = spikes_transform(spikes_in)
        else:
            spikes_in = encoder(x_b)  # ожидаем batch-encoder
            if torch.is_tensor(spikes_in) and spikes_in.device != device:
                spikes_in = spikes_in.to(device, non_blocking=True)

            # Conv-input path: accept [T,B,1,28,28] as-is.
            if torch.is_tensor(spikes_in) and spikes_in.dim() == 5:
                # expected [T,B,C,H,W]
                pass
            else:
                # Optional transform (e.g., flat->CHW).
                if spikes_transform is not None:
                    spikes_in = spikes_transform(spikes_in)

                # If still not conv-shaped, normalize to [T,B,784] for FC nets.
                if torch.is_tensor(spikes_in) and spikes_in.dim() == 5:
                    pass
                else:
                    if spikes_in.dim() == 4 and spikes_in.shape[2] == 1:   # [T,B,1,784]
                        spikes_in = spikes_in[:, :, 0, :]
                    elif spikes_in.dim() == 3:                              # [T,B,784]
                        pass
                    else:
                        raise RuntimeError(f"Unexpected spikes_in shape: {tuple(spikes_in.shape)}")

        # D) прогон
        # Ensure net and layers are on the target device (BindsNet can keep stale CUDA tensors).
        if hasattr(net, "to"):
            net.to(device)
        for l in getattr(net, "layers", {}).values():
            if hasattr(l, "to"):
                l.to(device)
        for c in getattr(net, "connections", {}).values():
            if hasattr(c, "to"):
                c.to(device)

        if step == 0:
            try:
                shp = tuple(spikes_in.shape) if hasattr(spikes_in, "shape") else type(spikes_in)
                lif_dev = None
                try:
                    lif_dev = getattr(lif_layer, "s").device if torch.is_tensor(getattr(lif_layer, "s", None)) else None
                except Exception:
                    lif_dev = None
                print(f"[collect_counts_plus_fast] spikes_in shape: {shp} | device={device} | lif.s.device={lif_dev}")
            except Exception:
                pass

        net.run(inputs={"Input": spikes_in}, time=T)

        # E) чтение и признаки
        s = mon.get("s")
        if s.dim() == 5:
            # Conv layer: [T,B,C,H,W] -> [T,B,N]
            Tt, Bb, Cc, Hh, Ww = s.shape
            s = s.view(Tt, Bb, Cc * Hh * Ww)
        elif s.dim() == 4 and s.shape[2] == 1:
            s = s[:, :, 0, :]
        elif s.dim() == 2:
            s = s[:, None, :]
        elif s.dim() == 3:
            pass
        else:
            raise RuntimeError(f"Unexpected monitor s shape: {tuple(s.shape)}")

        s = s.to(dtype=torch.float32, device=device)   # [T,B,N]
        counts_bn = s.sum(0)                           # [B,N]

        # WTA-гистограмма (векторно)
        active_tb = (s.sum(2) > 0)                     # [T,B]
        winners_tb = s.argmax(dim=2)                   # [T,B]
        cls_tb = lm[winners_tb]                        # [T,B]
        valid_tb = active_tb & (cls_tb >= 0)

        active_counts_b = active_tb.sum(0).to(torch.float32)  # [B]
        b_ids = torch.arange(B, device=device)[None, :].expand(T, B)
        flat_idx = (b_ids * 10 + cls_tb.clamp(min=0)).reshape(-1)
        w = valid_tb.to(torch.float32).reshape(-1)

        hist_flat = torch.bincount(flat_idx, weights=w, minlength=B * 10).to(torch.float32)
        hist_b10 = hist_flat.view(B, 10)

        denom = active_counts_b.clamp(min=1.0).unsqueeze(1)
        hist_b10 = hist_b10 / denom

        feats = torch.cat([counts_bn, hist_b10], dim=1)               # [B, N+10]

        # F) запись
        if use_memmap:
            # Write directly to disk-backed arrays.
            X[write_pos:write_pos + B, :] = feats.detach().cpu().numpy()
            y[write_pos:write_pos + B] = y_b.detach().cpu().numpy()
        else:
            if output_device.type == "cpu":
                X[write_pos:write_pos + B] = feats.detach().cpu()
                y[write_pos:write_pos + B] = y_b.detach().cpu()
            else:
                X[write_pos:write_pos + B] = feats.to(output_device)
                y[write_pos:write_pos + B] = y_b.to(output_device)

        write_pos += B

        # G) restore всё, что временно меняли
        net.reset_state_variables()
        mon.reset_state_variables()

        if vt_before is not None:
            _restore_vt()
            cur = getattr(lif_layer, vt_attr)
            if (write_pos < 5) or (debug_every and (write_pos % debug_every == 0)):
                cur_t = cur if torch.is_tensor(cur) else torch.as_tensor(float(cur), device=device)
                vt_after_restore_samples.append(float(cur_t.float().mean().detach().cpu()))
            if torch.is_tensor(cur):
                restored_ok += int(torch.allclose(cur, vt_backup, atol=1e-6))
            else:
                restored_ok += int(abs(float(cur) - float(vt_backup.item())) < 1e-6)

        if rec_conn is not None:
            with torch.no_grad():
                rec_conn.w.copy_(Wrec_backup)

        if progress and debug_every and (write_pos % debug_every == 0):
            sum_out = float(counts_bn.sum().detach().cpu())
            # iterator is tqdm only when progress bar is enabled.
            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(
                    f"sum_out={sum_out:.2f} vt({'abs' if threshold_abs is not None else 'off' if vt_eval_offset==0 else 'off='+str(vt_eval_offset)}) "
                    f"boost={encoder_rate_boost:g} inh={'off' if temp_disable_inh else 'on'}"
                )

    # 5) снять монитор
    net.monitors.pop("lif_tmp_counts_plus_fast", None)

    # 6) отладочная сводка
    avg_delta = float((delta_sum / max(1, delta_n)).detach().cpu()) if delta_n > 0 else 0.0
    debug_info = {
        "requested_offset": float(vt_eval_offset),
        "threshold_abs": float(threshold_abs) if threshold_abs is not None else None,
        "encoder_rate_boost": float(encoder_rate_boost),
        "temp_disable_inh": bool(temp_disable_inh),
        "applied_count": int(applied_cnt),
        "avg_delta_mean_vt": avg_delta,
        "restore_ok_fraction": (float(restored_ok / max(1, applied_cnt)) if applied_cnt > 0 else 1.0),
        "vt_before_samples": vt_before_samples,
        "vt_after_apply_samples": vt_after_apply_samples,
        "vt_after_restore_samples": vt_after_restore_samples,
        "device": str(device),
        "batch_size": int(batch_size),
        "write_pos": int(write_pos),
        "expected_batches": int((n_samples + batch_size - 1) // batch_size),
        "actual_batches": int(step + 1),
        "version": 3,
    }

    if verify_offset:
        if applied_cnt > 0:
            print(
                "[collect_counts_plus_cuda] offset summary:",
                f"requested={vt_eval_offset:+.3f}, applied={applied_cnt}, "
                f"avgΔvt={avg_delta:.3f}, restore_ok={restored_ok}/{applied_cnt}"
            )
        else:
            print(
                "[collect_counts_plus_cuda] offset summary:",
                f"requested={vt_eval_offset:+.3f}, applied=0, avgΔvt=0.000, restore_ok=–"
            )
    assert write_pos == n_samples, f"collected {write_pos}, expected {n_samples}"

    return (X, y, debug_info) if return_debug else (X, y)

