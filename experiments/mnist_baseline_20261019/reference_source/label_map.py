# label_map.py
from __future__ import annotations
from typing import Optional 
import torch
from bindsnet.network.monitors import Monitor
from bindsnet.datasets import MNIST
from torchvision import transforms
import os, time


# добавь в список экспорта:
__all__ = [
    # ... уже были ...
    "build_label_map",
    "save_label_map", "load_label_map",
]

def save_label_map(path: str, label_map: torch.Tensor, meta: Optional[dict] = None) -> None:
    """
    Save label_map tensor (long) + optional metadata as a torch file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "label_map": torch.as_tensor(label_map, dtype=torch.long).cpu(),
        "meta": {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {})},
    }
    torch.save(payload, path)
    print(f"[label_map] saved: {path}  shape={tuple(payload['label_map'].shape)}")

def load_label_map(path: str, device: Optional[str] = None) -> torch.Tensor:
    """
    Load label_map from disk. If device provided, move to that device.
    """
    payload = torch.load(path, map_location="cpu")
    lm = torch.as_tensor(payload["label_map"], dtype=torch.long)
    if device:
        lm = lm.to(device)
    print(f"[label_map] loaded: {path}  shape={tuple(lm.shape)}  meta={payload.get('meta',{})}")
    return lm

@torch.no_grad()
def build_label_map(net, input_layer, lif_layer, encoder, n_calib: int = 2000, T: int = 200, top_k: int = 3, seed: int = 123) -> torch.Tensor:
    for c in net.connections.values():
        if hasattr(c, "update_rule"):
            c.update_rule.nu = (torch.as_tensor(0.0), torch.as_tensor(0.0))

    lif_mon = Monitor(lif_layer, state_vars=("s",), time=T); net.add_monitor(lif_mon, name="lif_eval_tmp")

    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = MNIST(root="./data", train=True, download=True, transform=transform)
    idxs = list(range(min(n_calib, len(ds_train))))

    usage = torch.zeros((lif_layer.n,), dtype=torch.long)
    wins  = torch.zeros((lif_layer.n, 10), dtype=torch.long)

    # Ensure inputs are on the same device as the network runtime.
    # BindsNet will error if self.x and mask self.s are on different devices.
    try:
        dev = next(net.layers.values()).s.device  # type: ignore[attr-defined]
    except Exception:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in idxs:
        torch.manual_seed(seed + i)
        x = ds_train[i]["image"].to(dev)
        y = int(ds_train[i]["label"])
        spikes_in = encoder(x)
        if torch.is_tensor(spikes_in):
            spikes_in = spikes_in.to(dev)

        # Final safety: if BindsNet has stale CPU tensors inside layers, it can crash.
        # Re-apply net.to(dev) here (cheap on CPU, safe on CUDA).
        if hasattr(net, "to"):
            net.to(dev)

        net.run(inputs={"Input": spikes_in}, time=T)

        s = lif_mon.get("s")
        # FC: [T,1,N] -> [T,N]; Conv: [T,1,C,H,W] -> [T, C*H*W]
        if s.dim() == 5:
            s2 = s[:, 0, :, :, :].reshape(s.shape[0], -1)
        else:
            s2 = s[:, 0, :]
        counts = s2.sum(0)
        if counts.sum() > 0:
            k = min(top_k, lif_layer.n)
            topi = torch.topk(counts, k=k).indices
            for j in topi.tolist():
                usage[j] += 1; wins[j, y] += 1

        net.reset_state_variables(); lif_mon.reset_state_variables()

    net.monitors.pop("lif_eval_tmp", None)

    label_map = -torch.ones((lif_layer.n,), dtype=torch.long)
    active = (usage > 0).nonzero().flatten().tolist()
    for j in active:
        label_map[j] = wins[j].argmax().item()

    covered = int((label_map >= 0).sum())
    print(f"Label-map built: {covered}/{lif_layer.n} neurons assigned; active winners {int((usage>0).sum())}")
    return label_map
