"""U-Net for binary crack segmentation.

Written from scratch rather than imported, because the encoder/decoder
structure and the skip connections are the part worth understanding — and
the part an examiner will ask about.

Reference: Ronneberger, Fischer & Brox (2015), "U-Net: Convolutional
Networks for Biomedical Image Segmentation".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2, the basic block of every U-Net stage.

    BatchNorm was not in the original paper but is standard now; it makes
    training stable at the learning rates used here.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Halve the spatial size with max-pool, then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Double the spatial size, concatenate the skip connection, DoubleConv.

    Channel counts are passed explicitly rather than inferred, because the
    bilinear and transposed-convolution paths leave the upsampled tensor
    with different channel counts, and inferring it is a common shape bug.

    Args:
        in_ch:   channels arriving from the layer below.
        skip_ch: channels in the skip connection from the encoder.
        out_ch:  channels this stage should output.
        bilinear: True  -> parameter-free interpolation (fewer params,
                           no checkerboard artefacts).
                  False -> learned ConvTranspose2d. Worth an ablation.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            concat_ch = in_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            concat_ch = in_ch // 2 + skip_ch
        self.conv = DoubleConv(concat_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Odd input sizes leave the upsampled tensor a pixel short; pad to match.
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Binary-output U-Net.

    Returns raw logits, NOT probabilities. Every loss in src/losses.py
    expects logits, and inference applies the sigmoid itself. Keeping the
    sigmoid out of the model avoids the classic double-sigmoid bug.

    Channel plan for base=32, depth=4:
        encoder   3 -> 32 -> 64 -> 128 -> 256 -> 512 (bottleneck)
        decoder   512 -> 256 -> 128 -> 64 -> 32 -> 1

    Args:
        in_channels: 3 for RGB.
        base: channels at the first stage. 64 is the paper's value; 32
            halves memory and trains fine on a free Colab GPU.
        depth: number of downsampling stages.
        bilinear: see Up.
        dropout: applied at the bottleneck only.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base: int = 32,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.bilinear = bilinear

        # Encoder channel widths: [base, base*2, ..., base*2**depth]
        widths = [base * (2 ** i) for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            [Down(widths[i], widths[i + 1]) for i in range(depth)]
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Decoder mirrors the encoder: at stage i we come up from
        # widths[depth-i], meet skip widths[depth-i-1], output that width.
        self.ups = nn.ModuleList(
            [
                Up(
                    in_ch=widths[depth - i],
                    skip_ch=widths[depth - i - 1],
                    out_ch=widths[depth - i - 1],
                    bilinear=bilinear,
                )
                for i in range(depth)
            ]
        )

        self.outc = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.dropout(x)

        # skips[-1] is the bottleneck itself; walk back up through the rest.
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 2)])

        return self.outc(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning probabilities in [0, 1]."""
        return torch.sigmoid(self.forward(x))


def build_model(cfg: dict) -> UNet:
    """Construct a UNet from a config dict (see configs/default.yaml)."""
    m = cfg.get("model", {})
    return UNet(
        in_channels=m.get("in_channels", 3),
        base=m.get("base", 32),
        depth=m.get("depth", 4),
        bilinear=m.get("bilinear", True),
        dropout=m.get("dropout", 0.0),
    )


if __name__ == "__main__":
    # Shape checks across both upsampling modes and a non-power-of-two input.
    for bilinear in (True, False):
        net = None
        for size in (256, 320, 250):
            net = UNet(base=32, depth=4, bilinear=bilinear)
            dummy = torch.randn(2, 3, size, size)
            out = net(dummy)
            assert out.shape == (2, 1, size, size), (
                f"shape mismatch: bilinear={bilinear} size={size} -> {tuple(out.shape)}"
            )
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"bilinear={bilinear!s:<5}  params={n_params:,}  shapes OK")
    print("all shape checks passed")
