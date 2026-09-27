"""
H3 Residual Temporal Smooth — anti latent-flicker for H3 face refinement.

Drop this file into ComfyUI/custom_nodes/ and restart ComfyUI.

WHY
The MiniMax H3 video VAE packs 4 frames into one latent frame. Everything H3
decodes carries a texture/sharpness pulse with a period of exactly 4 frames
(6 Hz at 24 fps). On a small face it is invisible; once the face is regenerated
at 3x scale the pulse lives in eyes, lips and skin, and reads as trembling.

WHAT IT DOES
Takes the ORIGINAL crops (tracker output) and the REFINED crops (VAE decode),
splits refined = original + residual, and smooths ONLY the residual in time.
The source's own motion stays untouched and sharp; only what the model added
is stabilised. The default kernel [0.5, 1, 1, 1, 0.5] / 4 has exact zeros at
periods 4 and 2 frames, so the latent pulse is removed without a lag.

WIRING (inside the Face Pass subgraph)
  H3FaceTrackCrop.crops  -> original_crops
  VAEDecode (refined)    -> refined_crops
  images                 -> H3FaceStitch.refined_crops
"""

import torch


def _kernel(kind: str, sigma: float) -> torch.Tensor:
    base = torch.tensor([0.5, 1.0, 1.0, 1.0, 0.5], dtype=torch.float32) / 4.0
    if kind == "latent4_notch":
        return base
    if kind == "latent4_notch_wide":
        k = torch.nn.functional.conv1d(base.view(1, 1, -1), base.flip(0).view(1, 1, -1), padding=4).view(-1)
        return k / k.sum()
    r = max(1, int(round(sigma * 3)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _reflect_index(n: int, pad: int) -> torch.Tensor:
    idx = torch.arange(-pad, n + pad)
    period = 2 * (n - 1) if n > 1 else 1
    idx = torch.remainder(idx, period)
    return torch.where(idx >= n, period - idx, idx)


def _period4_share(series: torch.Tensor) -> float:
    x = series.double()
    if x.numel() < 24:
        return float("nan")
    trend = torch.nn.functional.avg_pool1d(x.view(1, 1, -1), 9, 1, 4, count_include_pad=False).view(-1)
    x = (x - trend)[5:-5]
    win = torch.hann_window(x.numel(), dtype=torch.float64)
    p = torch.fft.rfft(x * win).abs() ** 2
    f = torch.fft.rfftfreq(x.numel(), dtype=torch.float64)
    p, f = p[1:], f[1:]
    per = 1.0 / f
    band = (per > 3.6) & (per < 4.4)
    return float(p[band].sum() / p.sum())


class H3ResidualTemporalSmooth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_crops": ("IMAGE",),
                "refined_crops": ("IMAGE",),
                "kernel": (["latent4_notch", "latent4_notch_wide", "gaussian"], {"default": "latent4_notch"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "gaussian_sigma": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 6.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "H3 FaceRefine/extra"

    def run(self, original_crops, refined_crops, kernel, strength, gaussian_sigma):
        o, r = original_crops, refined_crops
        if o.shape != r.shape:
            raise ValueError(
                f"original_crops {tuple(o.shape)} and refined_crops {tuple(r.shape)} must match "
                "(same frame count and canvas size)."
            )
        T, H, W, C = r.shape
        k = _kernel(kernel, gaussian_sigma)
        pad = (k.numel() - 1) // 2
        if T <= pad:
            return (r, f"skipped: only {T} frames")
        src = _reflect_index(T, pad)
        out = torch.empty_like(r)
        before = torch.zeros(T, dtype=torch.float64)
        after = torch.zeros(T, dtype=torch.float64)
        rows = max(1, int(64 * 1024 / max(W, 1)))
        for y0 in range(0, H, rows):
            y1 = min(H, y0 + rows)
            oo = o[:, y0:y1].float()
            res = r[:, y0:y1].float() - oo
            padded = res[src.to(res.device)]
            sm = torch.zeros_like(res)
            for j in range(k.numel()):
                sm += padded[j:j + T] * float(k[j])
            mix = sm * strength + res * (1.0 - strength)
            out[:, y0:y1] = (oo + mix).clamp(0.0, 1.0).to(r.dtype)
            before += res.abs().mean(dim=(1, 2, 3)).double().cpu() * (y1 - y0)
            after += mix.abs().mean(dim=(1, 2, 3)).double().cpu() * (y1 - y0)
        jb = float((before[1:] - before[:-1]).std())
        ja = float((after[1:] - after[:-1]).std())
        cut = (1.0 - ja / jb) * 100.0 if jb > 0 else 0.0
        report = (
            f"frames={T} canvas={W}x{H} kernel={kernel} strength={strength:.2f}\n"
            f"4-frame pulse share in model change: {_period4_share(before):.2f} -> {_period4_share(after):.2f}\n"
            f"frame-to-frame jitter of model change: -{cut:.0f}%"
        )
        return (out, report)


NODE_CLASS_MAPPINGS = {"H3ResidualTemporalSmooth": H3ResidualTemporalSmooth}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ResidualTemporalSmooth": "H3 Residual Temporal Smooth (anti latent-flicker)"}
