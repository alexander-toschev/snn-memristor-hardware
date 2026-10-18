
# snn_mnist_net.py
from __future__ import annotations

import os
import random
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torchvision import transforms

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import PostPre

from encoders import LatencyEncoder, PoissonEncoder

try:
    from torchvision import transforms as _tv_transforms
    PREPROCESS = _tv_transforms.Compose([
        _tv_transforms.Grayscale(num_output_channels=1),
        _tv_transforms.Resize((28, 28), antialias=True),
        _tv_transforms.ToTensor(),
    ])
except Exception:
    PREPROCESS = None

__all__ = [
    "Cfg", "set_seed", "build_net", "build_encoder_from_cfg",
    "attach_monitors", "SNNMeter", "apply_wta", "activity_metrics", "run_experiment",
    "print_lif_params", "tune_lif_params", "save_snn", "load_weights_into", "load_snn_cfg"
]


@dataclass
class Cfg:
    time: int = 200
    n_hidden: int = 100
    N: int = 200
    seed: int = 42
    device: str = "cpu"
    log_every: int = 50
    debug: bool = False

    nu_plus: float = 1e-4
    nu_minus: float = -1e-3

    enable_inhibition_at_start: bool = True
    inhib_strength: float = 0.705

    encoder: str = "poisson"
    poisson_rate_scale: float = 0.011
    poisson_base_seed: int = 123
    poisson_deterministic: bool = False
    encoder_out_format: str = "auto"
    encoder_rate_boost: float = 3.0

    thresh_init: float = 0.38
    thresh_min: float = 0.15
    thresh_max: float = 1.20
    target_spikes: float = 1.5
    ema_alpha: float = 0.9
    ema_k: float = 0.02

    tau_val: float = 150.0
    refrac_val: float = 2.0
    reset_val: float = 0.0
    rest_val: float = 0.0

    w_init_lo: float = 0.25
    w_init_hi: float = 0.8
    w_clip_min: float = 0.0
    w_clip_max: float = 1.0
    wmin: float = 0.0
    wmax: float = 1.0

    warmup_N: int = 10000
    top_k: int = 0

    latency_x_min: float = 0.05

    strong_inh_matrix: bool = True
    strong_inh_value: float = 0.65

    bootstrap_threshold_enable: bool = True
    bootstrap_threshold: float = 0.12
    bootstrap_refrac: float = 2.0

    def torch_device(self) -> torch.device:
        dev = torch.device(self.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return dev


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_str_or_device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def _layer_device(layer, default: str = "cpu") -> torch.device:
    for name in ("v_thresh", "thresh", "v", "s", "x", "rest", "reset", "refrac", "tc_decay", "tau"):
        x = getattr(layer, name, None)
        if torch.is_tensor(x):
            return x.device
    return torch.device(default)


def _set_tensor_attr(obj, name: str, value):
    setattr(obj, name, value)


def _move_layer_state_(layer, device: torch.device) -> None:
    # NOTE: BindsNet layers can keep some runtime scalars/tensors (e.g. dt) on CPU.
    # That can later cause device-mismatch errors (e.g. masked_fill_) when the layer
    # state is on CUDA. We proactively move all known state tensors to the target device.
    for name in (
        "s", "x", "v", "refrac_count", "trace", "summed",
        "rest", "v_rest", "reset", "v_reset", "refrac",
        "thresh", "v_thresh", "tc_decay", "tau",
        "dt",
    ):
        x = getattr(layer, name, None)
        if torch.is_tensor(x):
            _set_tensor_attr(layer, name, x.to(device))


def _move_connection_state_(conn: Connection, device: torch.device) -> None:
    for name in ("w", "b", "wmin", "wmax"):
        x = getattr(conn, name, None)
        if torch.is_tensor(x):
            _set_tensor_attr(conn, name, x.to(device))
    if hasattr(conn, "update_rule") and conn.update_rule is not None:
        nu = getattr(conn.update_rule, "nu", None)
        if isinstance(nu, tuple):
            conn.update_rule.nu = tuple(v.to(device) if torch.is_tensor(v) else v for v in nu)


def _move_net_runtime_(net: Network, input_layer: Input, lif_layer: LIFNodes,
                       connection: Connection, recurrent_inh: Optional[Connection],
                       device: torch.device) -> None:
    if hasattr(net, "to"):
        net.to(device)
    _move_layer_state_(input_layer, device)
    _move_layer_state_(lif_layer, device)
    _move_connection_state_(connection, device)
    if recurrent_inh is not None:
        _move_connection_state_(recurrent_inh, device)


def _layer_thresh(layer: LIFNodes) -> Optional[Tensor]:
    return getattr(layer, "v_thresh", getattr(layer, "thresh", None))


def _set_layer_thresh(layer: LIFNodes, vt: Tensor) -> None:
    if hasattr(layer, "v_thresh"):
        layer.v_thresh = vt
    else:
        layer.thresh = vt


def build_encoder_from_cfg(cfg: Cfg):
    enc = (cfg.encoder or "poisson").lower()
    out_fmt = getattr(cfg, "encoder_out_format", "auto")

    if enc == "poisson":
        return PoissonEncoder(
            T=cfg.time,
            rate_scale=cfg.poisson_rate_scale,
            base_seed=cfg.poisson_base_seed,
            deterministic=bool(getattr(cfg, "poisson_deterministic", False)),
            out_format=out_fmt,
        )

    if enc == "latency":
        return LatencyEncoder(
            time=cfg.time,
            out_format=out_fmt,
            x_min=float(getattr(cfg, "latency_x_min", 0.0)),
        )

    raise ValueError(f"Unknown encoder: {cfg.encoder}")


def tune_lif_params(lif: LIFNodes, n_hidden: int, cfg: Cfg, device: torch.device | str | None = None) -> None:
    dev = _device_from_str_or_device(device) if device is not None else _layer_device(lif)
    with torch.no_grad():
        vt = torch.full((n_hidden,), float(cfg.thresh_init), device=dev, dtype=torch.float32)
        _set_layer_thresh(lif, vt)

        if hasattr(lif, "tc_decay"):
            lif.tc_decay = torch.tensor(float(cfg.tau_val), device=dev, dtype=torch.float32)
        elif hasattr(lif, "tau"):
            lif.tau = torch.tensor(float(cfg.tau_val), device=dev, dtype=torch.float32)

        if hasattr(lif, "refrac"):
            lif.refrac = torch.tensor(float(cfg.refrac_val), device=dev, dtype=torch.float32)

        if hasattr(lif, "v_reset"):
            lif.v_reset = torch.tensor(float(cfg.reset_val), device=dev, dtype=torch.float32)
        elif hasattr(lif, "reset"):
            lif.reset = torch.tensor(float(cfg.reset_val), device=dev, dtype=torch.float32)

        if hasattr(lif, "rest"):
            lif.rest = torch.tensor(float(cfg.rest_val), device=dev, dtype=torch.float32)
        elif hasattr(lif, "v_rest"):
            lif.v_rest = torch.tensor(float(cfg.rest_val), device=dev, dtype=torch.float32)


def print_lif_params(lif: LIFNodes) -> None:
    def _stat(x):
        if x is None:
            return "None"
        if torch.is_tensor(x):
            if x.numel() == 1:
                return f"scalar {x.item():.3f} @ {x.device}"
            return (
                f"tensor mean={x.float().mean():.3f} std={x.float().std():.3f} "
                f"shape={tuple(x.shape)} @ {x.device}"
            )
        return str(x)

    vt = getattr(lif, "v_thresh", getattr(lif, "thresh", None))
    tau = getattr(lif, "tc_decay", getattr(lif, "tau", None))
    refr = getattr(lif, "refrac", None)
    reset = getattr(lif, "v_reset", getattr(lif, "reset", None))
    rest = getattr(lif, "rest", getattr(lif, "v_rest", None))
    dt = getattr(lif, "dt", None)

    print(f"[lif] thresh = {_stat(vt)}")
    print(f"[lif] tc_decay = {_stat(tau)}")
    print(f"[lif] refrac = {_stat(refr)}")
    print(f"[lif] reset = {_stat(reset)}")
    print(f"[lif] rest  = {_stat(rest)}")
    if dt is not None:
        print(f"[lif] dt = {_stat(dt)}")


def build_net(cfg: Cfg) -> Tuple[Network, Input, LIFNodes, Connection, Optional[Connection], Tensor]:
    device = cfg.torch_device()

    # Build network on CPU first.
    net = Network()
    input_layer = Input(n=784, traces=True)
    lif_layer = LIFNodes(n=cfg.n_hidden, traces=True)

    net.add_layer(input_layer, name="Input")
    net.add_layer(lif_layer, name="LIF")

    connection = Connection(source=input_layer, target=lif_layer, wmin=cfg.wmin, wmax=cfg.wmax)
    connection.update_rule = PostPre(
        connection=connection,
        nu=(torch.tensor(float(cfg.nu_plus)), torch.tensor(float(cfg.nu_minus))),
    )
    net.add_connection(connection, source="Input", target="LIF")

    W_inh = torch.full((cfg.n_hidden, cfg.n_hidden), -float(cfg.inhib_strength), dtype=torch.float32)
    W_inh.fill_diagonal_(0.0)
    recurrent_inh = Connection(source=lif_layer, target=lif_layer, w=W_inh.clone(), wmin=-1.0, wmax=0.0)
    net.add_connection(recurrent_inh, source="LIF", target="LIF")

    # Move everything only after the network is fully assembled.
    _move_net_runtime_(net, input_layer, lif_layer, connection, recurrent_inh, device)

    # Initialize params on target device.
    tune_lif_params(lif_layer, cfg.n_hidden, cfg, device=device)

    with torch.no_grad():
        connection.w.data.uniform_(float(cfg.w_init_lo), float(cfg.w_init_hi))
        connection.w.data.clamp_(float(cfg.w_clip_min), float(cfg.w_clip_max))

        if cfg.strong_inh_matrix:
            dev = recurrent_inh.w.device
            dtype = recurrent_inh.w.dtype
            I = torch.eye(lif_layer.n, device=dev, dtype=dtype)
            if torch.is_tensor(recurrent_inh.wmin):
                recurrent_inh.wmin.data.fill_(-1.0)
            recurrent_inh.w.copy_(-float(cfg.strong_inh_value) * (1 - I))

        if cfg.bootstrap_threshold_enable:
            vt0 = torch.full((lif_layer.n,), float(cfg.bootstrap_threshold), device=device, dtype=torch.float32)
            _set_layer_thresh(lif_layer, vt0)
            if hasattr(lif_layer, "refrac"):
                lif_layer.refrac = torch.tensor(float(cfg.bootstrap_refrac), device=device, dtype=torch.float32)

        if not cfg.enable_inhibition_at_start:
            recurrent_inh.w.zero_()

    _move_net_runtime_(net, input_layer, lif_layer, connection, recurrent_inh, device)
    print_lif_params(lif_layer)
    return net, input_layer, lif_layer, connection, recurrent_inh, W_inh


def attach_monitors(net: Network, X: Input, H: LIFNodes, T: int) -> Dict[str, Monitor]:
    mon_X = Monitor(obj=X, state_vars=("s",), time=T)
    net.add_monitor(mon_X, name="mon_X")
    mon_H = Monitor(obj=H, state_vars=("s", "v"), time=T)
    net.add_monitor(mon_H, name="mon_H")
    return {"mon_X": mon_X, "mon_H": mon_H}


def _to_2d(spikes: Tensor) -> Tensor:
    if spikes.ndim == 3:
        return spikes[:, 0, :]
    return spikes


def apply_wta(spikes_H: Tensor, top_k: int = 1):
    s2 = _to_2d(spikes_H)
    per_neuron = s2.sum(dim=0)
    if per_neuron.sum() == 0:
        return False, None
    k = min(top_k, per_neuron.numel())
    _, idxs = torch.topk(per_neuron, k=k)
    spikes_H.zero_()
    for j in idxs.tolist():
        if spikes_H.ndim == 3:
            spikes_H[:, 0, j] = 1.0
        else:
            spikes_H[:, j] = 1.0
    return True, idxs.tolist()


class SNNMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples = 0
        self.S_out = 0
        self.S_in = 0
        self.SynOps = 0
        self.V_updates = 0
        self.usage_counts: Dict[int, int] = {}

    def log_sample(self, lif_s: Tensor, in_s: Tensor, n_hidden: int, T: int, winners=None) -> None:
        lif2 = _to_2d(lif_s)
        in2 = _to_2d(in_s)
        self.S_out += int(lif2.sum().item())
        self.S_in += int(in2.sum().item())
        self.SynOps += int(in2.sum().item()) * n_hidden
        self.V_updates += n_hidden * T
        self.samples += 1
        if winners:
            for j in winners:
                self.usage_counts[j] = self.usage_counts.get(j, 0) + 1

    def report(self, a: float = 1.0, b: float = 0.05, c: float = 0.005) -> Dict[str, float]:
        s = max(1, self.samples)
        HHI_win = 0.0
        if self.usage_counts:
            import numpy as np
            tot = sum(self.usage_counts.values())
            ps = np.array([v / tot for v in self.usage_counts.values()], dtype=float)
            HHI_win = float((ps ** 2).sum())
        return {
            "spikes_per_sample": self.S_out / s,
            "synops_per_sample": self.SynOps / s,
            "v_updates_per_sample": self.V_updates / s,
            "energy_proxy_per_sample": (a * self.S_out + b * self.SynOps + c * self.V_updates) / s,
            "winners_unique": len(self.usage_counts),
            "winner_HHI": HHI_win,
        }


@torch.no_grad()
def activity_metrics(mon_H: Monitor) -> Dict[str, float]:
    s: Tensor = mon_H.get("s")
    T, B, H = s.shape
    spikes_per_sample = s.sum(dim=(0, 2)).mean().item()
    per_sample_counts = s.sum(dim=0)
    winners = per_sample_counts.argmax(dim=1)
    winners_unique = winners.unique().numel()
    hist = torch.bincount(winners, minlength=H).float()
    p = (hist / max(1, hist.sum())).clamp_min(1e-12)
    hhi = float((p ** 2).sum().item())
    energy_proxy_per_sample = float(s.sum().item() / max(1, B))
    return {
        "spikes_per_sample": float(spikes_per_sample),
        "winners_unique": int(winners_unique),
        "winner_HHI": float(hhi),
        "energy_proxy_per_sample": energy_proxy_per_sample,
    }


class _ThreshEMA:
    def __init__(self):
        self.rate_ema: Optional[Tensor] = None

    @torch.no_grad()
    def step(self, layer: LIFNodes, spike_counts: Tensor, T: int, cfg: Cfg) -> None:
        vt = _layer_thresh(layer)
        if vt is None:
            return

        dev = vt.device
        dtype = vt.dtype

        rate = spike_counts.to(device=dev, dtype=dtype) / max(1, T)

        if self.rate_ema is None:
            self.rate_ema = rate.clone()
        else:
            if self.rate_ema.device != dev or self.rate_ema.dtype != dtype:
                self.rate_ema = self.rate_ema.to(device=dev, dtype=dtype)
            self.rate_ema = cfg.ema_alpha * self.rate_ema + (1.0 - cfg.ema_alpha) * rate

        target_per_step = torch.tensor(
            float(cfg.target_spikes) / max(1, T),
            device=dev,
            dtype=dtype,
        )

        vt_new = vt + cfg.ema_k * (self.rate_ema - target_per_step)
        vt_new.clamp_(float(cfg.thresh_min), float(cfg.thresh_max))
        _set_layer_thresh(layer, vt_new)


def _set_stdp_nu(conn: Connection, nu_p: float, nu_m: float) -> None:
    if hasattr(conn, "update_rule") and conn.update_rule is not None:
        dev = conn.w.device
        conn.update_rule.nu = (
            torch.tensor(float(nu_p), device=dev),
            torch.tensor(float(nu_m), device=dev),
        )


def _print_banner(cfg: Cfg) -> None:
    print("\nSNN-MNIST TRAIN START")
    print(f"seed={cfg.seed}  device={cfg.device}  T={cfg.time}  N={cfg.N}")
    print(f"n_hidden={cfg.n_hidden}  thresh_init={cfg.thresh_init}")
    print(f"STDP: nu_plus={cfg.nu_plus}  nu_minus={cfg.nu_minus}")
    print(f"FF init: w∈[{cfg.w_init_lo}, {cfg.w_init_hi}], clip=[{cfg.w_clip_min},{cfg.w_clip_max}]")
    print(f"LIF: rest_val={cfg.rest_val}  reset_val={cfg.reset_val}  tau={cfg.tau_val}  refrac={cfg.refrac_val}")
    print(f"Encoder={cfg.encoder}  poisson_rate_scale={cfg.poisson_rate_scale}  base_seed={cfg.poisson_base_seed}")
    print(f"Inhibition: enable={cfg.enable_inhibition_at_start}  inhib_strength={cfg.inhib_strength}  top_k={cfg.top_k}")
    print("---------------------------\n")


def _maybe_tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _prepare_spikes(spikes: Tensor, device: torch.device) -> Tensor:
    if not torch.is_tensor(spikes):
        spikes = torch.as_tensor(spikes)
    return spikes.to(device=device, dtype=torch.float32, non_blocking=True)


def _post_reset_realign(net: Network, input_layer: Input, lif_layer: LIFNodes,
                        connection: Connection, recurrent_inh: Optional[Connection],
                        device: torch.device) -> None:
    net.reset_state_variables()
    _move_net_runtime_(net, input_layer, lif_layer, connection, recurrent_inh, device)


def run_experiment(cfg: Cfg, verbose: bool = True, progress: bool = True):
    set_seed(cfg.seed)
    device = cfg.torch_device()
    print("Используем:", device)
    _print_banner(cfg)

    from bindsnet.datasets import MNIST

    transform = transforms.Compose([transforms.ToTensor()])
    ds = MNIST(root="./data", train=True, download=True, transform=transform)
    n_train = min(cfg.N, len(ds))

    net, input_layer, lif_layer, connection, recurrent_inh, W_inh = build_net(cfg)
    enc = build_encoder_from_cfg(cfg)
    mons = attach_monitors(net, input_layer, lif_layer, cfg.time)

    meter = SNNMeter()
    ema = _ThreshEMA()

    if cfg.warmup_N > 0:
        _set_stdp_nu(connection, 0.0, 0.0)
        wbar = _maybe_tqdm(
            range(min(cfg.warmup_N, n_train)),
            desc=f"Warmup (T={cfg.time})",
            ncols=100,
            disable=not progress,
        )
        for wi in wbar:
            x = ds[wi]["image"].to(device)
            spikes = _prepare_spikes(enc(x), device)

            if wi < 3:
                print(f"[enc] shape={tuple(spikes.shape)} sum={float(spikes.sum())} max={float(spikes.max())}")
                if spikes.ndim == 3:
                    per_t = spikes[:, 0, :].sum(dim=1)
                elif spikes.ndim == 4:
                    per_t = spikes[:, 0, 0, :].sum(dim=1)
                else:
                    raise ValueError(f"Unexpected spikes ndim={spikes.ndim}")
                tmax = int(per_t.argmax().item())
                print(f"[enc] per_t: min={float(per_t.min())} max={float(per_t.max())} t_argmax={tmax}")

            net.run(inputs={"Input": spikes}, time=cfg.time)

            in_s = mons["mon_X"].get("s")
            lif_s = mons["mon_H"].get("s")
            v = mons["mon_H"].get("v")

            if wi < 3:
                vt = _layer_thresh(lif_layer)
                print(
                    f"[mon] in_sum={float(in_s.sum())} lif_sum={float(lif_s.sum())} "
                    f"v_max={float(v.max())} vt_min={float(vt.min()) if vt is not None else None}"
                )

            sc = (lif_s[:, 0, :] if lif_s.ndim == 3 else lif_s).sum(dim=0).float()
            ema.step(lif_layer, sc, cfg.time, cfg)

            with torch.no_grad():
                connection.w.clamp_(float(cfg.w_clip_min), float(cfg.w_clip_max))

            _post_reset_realign(net, input_layer, lif_layer, connection, recurrent_inh, device)
            for m in mons.values():
                m.reset_state_variables()

            if progress:
                rate = float((sc / max(1, cfg.time)).mean().item())
                wbar.set_postfix_str(f"avg_rate={rate:.3f} target={cfg.target_spikes:.2f}")

            if wi < 3:
                w = connection.w
                print(f"[W] mean={float(w.mean()):.6f} max={float(w.max()):.6f} min={float(w.min()):.6f}")

    _set_stdp_nu(connection, cfg.nu_plus, cfg.nu_minus)
    pbar = _maybe_tqdm(
        range(n_train),
        desc=f"Train (N={n_train}, T={cfg.time})",
        ncols=100,
        disable=not progress,
    )

    for i in pbar:
        x = ds[i]["image"].to(device)
        spikes = _prepare_spikes(enc(x), device)

        if i < 3:
            print(f"[enc] shape={tuple(spikes.shape)} sum={float(spikes.sum())} max={float(spikes.max())}")
            if spikes.ndim == 3:
                per_t = spikes[:, 0, :].sum(dim=1)
            elif spikes.ndim == 4:
                per_t = spikes[:, 0, 0, :].sum(dim=1)
            else:
                raise ValueError(f"Unexpected spikes ndim={spikes.ndim}")
            tmax = int(per_t.argmax().item())
            print(f"[enc] per_t: min={float(per_t.min())} max={float(per_t.max())} t_argmax={tmax}")

        net.run(inputs={"Input": spikes}, time=cfg.time)

        lif_s = mons["mon_H"].get("s")
        in_s = mons["mon_X"].get("s")

        winners = []
        if cfg.top_k and cfg.top_k > 0:
            ok, idxs = apply_wta(lif_s, top_k=cfg.top_k)
            if ok and idxs is not None:
                winners = idxs

        sc = (lif_s[:, 0, :] if lif_s.ndim == 3 else lif_s).sum(dim=0).float()
        ema.step(lif_layer, sc, cfg.time, cfg)

        with torch.no_grad():
            connection.w.clamp_(float(cfg.w_clip_min), float(cfg.w_clip_max))

        meter.log_sample(lif_s, in_s, cfg.n_hidden, cfg.time, winners=winners)

        _post_reset_realign(net, input_layer, lif_layer, connection, recurrent_inh, device)
        for m in mons.values():
            m.reset_state_variables()

        if verbose and ((i + 1) % max(1, cfg.log_every) == 0):
            rpt_mid = meter.report()
            report_string = (
                f"[{i+1:5d}/{n_train}] spikes/sample={rpt_mid['spikes_per_sample']:.2f} "
                f"uniq={rpt_mid['winners_unique']} HHI={rpt_mid['winner_HHI']:.3f} "
                f"energy≈{rpt_mid['energy_proxy_per_sample']:.1f}"
            )
            if not progress:
                print(report_string)

        if i < 3:
            w = connection.w
            print(f"[W] mean={float(w.mean()):.6f} max={float(w.max()):.6f} min={float(w.min()):.6f}")

        if verbose and progress:
            rpt_mid = meter.report()
            pbar.set_postfix_str(
                f"spikes={rpt_mid['spikes_per_sample']:.2f} "
                f"uniq={rpt_mid['winners_unique']} "
                f"HHI={rpt_mid['winner_HHI']:.3f} "
                f"e≈{rpt_mid['energy_proxy_per_sample']:.1f}"
            )

    rpt = meter.report()
    out = {**asdict(cfg), **rpt}
    return out, connection, lif_layer, net, enc


def save_snn(
    path: str,
    cfg: Cfg,
    connection: Connection,
    lif_layer: LIFNodes,
    train_summary: Optional[Dict] = None,
    eval_summary: Optional[Dict] = None,
    notes: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    vt = _layer_thresh(lif_layer)
    if vt is None:
        raise ValueError("Could not find layer threshold tensor")

    W = connection.w.detach().cpu()
    vt_cpu = vt.detach().cpu()

    weight_stats = {
        "W_shape": tuple(W.shape),
        "W_min": float(W.min().item()) if W.numel() else None,
        "W_max": float(W.max().item()) if W.numel() else None,
        "v_thresh_shape": tuple(vt_cpu.shape),
        "v_thresh_min": float(vt_cpu.min().item()) if vt_cpu.numel() else None,
        "v_thresh_max": float(vt_cpu.max().item()) if vt_cpu.numel() else None,
    }

    ckpt = {
        "schema_version": 2,
        "saved_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cfg": asdict(cfg),
        "W": W,
        "v_thresh": vt_cpu,
        "weight_stats": weight_stats,
    }

    if train_summary is not None:
        ckpt["train_summary"] = train_summary
    if eval_summary is not None:
        ckpt["eval_summary"] = eval_summary
    if notes is not None:
        ckpt["notes"] = notes

    torch.save(ckpt, path)
    print(f"Saved to {path} | W {tuple(W.shape)} | v_thresh {tuple(vt_cpu.shape)}")


def update_snn_ckpt(path: str, **fields) -> None:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Чекпойнт должен быть dict")

    payload.setdefault("schema_version", 2)
    payload["updated_at_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for k, v in fields.items():
        payload[k] = v

    torch.save(payload, path)
    print(f"Updated ckpt: {path} (fields: {list(fields.keys())})")


def load_snn_summaries(path: str) -> Dict[str, Optional[Dict]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return {"train_summary": None, "eval_summary": None, "notes": None}
    return {
        "train_summary": payload.get("train_summary"),
        "eval_summary": payload.get("eval_summary"),
        "notes": payload.get("notes"),
    }


def load_weights_into(net: Network, connection: Connection, lif_layer: LIFNodes, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vt = _layer_thresh(lif_layer)
    if vt is None:
        raise ValueError("Could not find layer threshold tensor")

    with torch.no_grad():
        connection.w.copy_(ckpt["W"].to(connection.w.device))
        _set_layer_thresh(lif_layer, ckpt["v_thresh"].to(vt.device))
    print(f"Loaded from {ckpt_path}")


def load_snn_cfg(ckpt_path: str) -> Cfg:
    payload = torch.load(ckpt_path, map_location="cpu")
    return Cfg(**payload["cfg"])
