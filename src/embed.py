"""Frozen V-JEPA 2 video features: one vector per timestep from a sliding window of frames ending at t."""
import torch
import torch.nn.functional as F
from transformers import VJEPA2Model

MODEL = "facebook/vjepa2-vitl-fpc64-256"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def load(device="cuda"):
    return VJEPA2Model.from_pretrained(MODEL, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()


@torch.inference_mode()
def embed(model, video, window=16, stride=1, batch=8, size=256):
    """video: (T, H, W, 3) uint8 array, already cropped to the region you want, upright.
    Returns (ceil(T/stride), 1024) float32: mean-pooled encoder tokens of frames [t-window+1, t];
    the start is padded by repeating frame 0. Timestep t corresponds to row t // stride."""
    device, dtype = model.device, model.dtype
    x = torch.as_tensor(video).permute(0, 3, 1, 2).float().div(255)
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)  # (T, 3, s, s)
    x = torch.cat([x[:1].expand(window - 1, -1, -1, -1), x])  # pad so every t has a full window
    ts = list(range(0, len(video), stride))
    out = []
    for i in range(0, len(ts), batch):
        clips = torch.stack([x[t:t + window] for t in ts[i:i + batch]])  # (B, window, 3, s, s)
        clips = ((clips - MEAN) / STD).to(device, dtype)
        h = model(pixel_values_videos=clips, skip_predictor=True).last_hidden_state
        out.append(h.mean(1).float().cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":  # smoke test: shapes, and a static video should give near-identical features over time
    import numpy as np
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    m = load(dev)
    vid = np.random.default_rng(0).integers(0, 255, (1, 120, 160, 3), dtype=np.uint8).repeat(20, 0)
    e = embed(m, vid, stride=5)
    assert e.shape == (4, 1024), e.shape
    cos = (e / np.linalg.norm(e, axis=1, keepdims=True)) @ (e[0] / np.linalg.norm(e[0]))
    assert cos.min() > 0.99, cos
    print("ok", e.shape, cos)
