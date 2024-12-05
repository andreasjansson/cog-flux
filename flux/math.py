from pathlib import Path
import math
import torch
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
import matplotlib.pyplot as plt


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attention_mask: Tensor | None = None, viz_path: Path | None = None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    if viz_path is not None:
        visualize_attention(q, k, v, viz_path)

    # Only enable flash attention backend
    if attention_mask is None:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def visualize_attention(q: Tensor, k: Tensor, v: Tensor, viz_path: Path):
    return
    with torch.no_grad():
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        for i in range(20):
            viz_path_i = viz_path.parent / (viz_path.name + f"-{i}.png")

            print(f"Writing {viz_path_i}")

            #print(attn_weights.sum([0, 1])[:256].mean(1)[:8])
            redux_viz = attn_weights.sum([0, 1])[i][256:].reshape([64,64])

            plt.figure(figsize=(5, 5))
            plt.imshow(redux_viz.to(torch.float32).cpu().numpy())
            plt.colorbar()

            plt.savefig(viz_path_i)
            plt.close()


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
