import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow, grid):
        new_locs = grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    def __init__(self, nsteps):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer()

    def forward(self, vec, grid):
        deforms = []
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec, grid)
            deforms.append(vec)
        return deforms


class MorphPatchEmb2D(nn.Module):
    def __init__(self, in_ch, embed_dim, nsteps, kernel_size, patch_size):
        super().__init__()
        self.vec_conv = nn.Conv2d(in_ch, 2 * in_ch, kernel_size, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(2 * in_ch)

        self.VecInt = VecInt(nsteps)
        self.transformer = SpatialTransformer()

        self.fuse_conv = nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=(nsteps, 1, 1))

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_ch,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, f):
        device = f.device

        vec = self.vec_conv(f)
        vec = self.bn(vec)

        bs, c, h, w = vec.shape
        vec = vec.view(bs * c // 2, 2, h, w)

        # build grid
        vectors = [torch.arange(0, s) for s in [h, w]]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids).unsqueeze(0).float().to(device)

        deforms = self.VecInt(vec, grid)
        finalmaps = []

        for j in range(len(deforms)):
            finalmap = self.transformer(f.view(bs * c // 2, 1, h, w), deforms[j], grid)
            finalmap = finalmap.view(bs, c // 2, h, w)
            finalmaps.append(finalmap)

        stacked = torch.stack(finalmaps, dim=0).permute(1, 2, 0, 3, 4)
        fused = self.fuse_conv(stacked).squeeze(2)

        embeddings = self.patch_embeddings(fused)
        return embeddings


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":

    batch = 2
    in_ch = 8
    H = W = 64
    nsteps = 4
    kernel_size = 3
    patch_size = 4
    embed_dim = 128

    model = MorphPatchEmb2D(in_ch, embed_dim, nsteps, kernel_size, patch_size).cuda()
    x = torch.randn(batch, in_ch, H, W).cuda()

    out = model(x)
    
    print("Input:", x.shape)
    print("Output:", out.shape)
