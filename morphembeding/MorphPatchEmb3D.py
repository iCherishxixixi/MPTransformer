import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import PatchEmbed

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()

        self.mode = mode
    def forward(self, src, flow, grid):
        # new locations
        new_locs = grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer()

    def forward(self, vec, grid):
        deforms = []
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec, grid)
            deforms.append(vec)
        return deforms


class MorphPatchEmb3D(nn.Module):
    def __init__(self, in_ch, embed_dim, nsteps, patch_size, kernel_size):
        super(MorphPatchEmb3D, self).__init__()
        self.vec_conv = nn.Conv3d(in_ch, 3 * in_ch, kernel_size, padding=(kernel_size - 1) // 2)        
        self.bn = nn.BatchNorm3d(3 * in_ch)
        
        self.VecInt = VecInt(nsteps)
        self.transformer = SpatialTransformer()
        
        self.patch_embed = PatchEmbed(
                                      patch_size=patch_size,
                                      in_chans=nsteps*in_ch,
                                      embed_dim=embed_dim,
                                      spatial_dims=3
                                      )

    def forward(self, f):
        device = f.device
        
        vec = self.vec_conv(f)
        vec = self.bn(vec)
        # offset = torch.tanh(offset)
        bs, c, l, h, w = vec.shape
        vec = vec.view(bs*c//3, 3, l, h, w)
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in [l, h, w]]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).requires_grad_(False).to(device)
        
        deforms = self.VecInt(vec, grid)
        finalmaps = []
        
        for j in range(len(deforms)):
            finalmap = self.transformer(f.view(bs*c//3, 1, l, h, w), deforms[j], grid)
            finalmap = finalmap.view(bs, c//3, l, h, w)
            finalmaps.append(finalmap)
            
        dflsf = torch.stack(finalmaps, dim=1).squeeze(2)
        embedings = self.patch_embed(dflsf)
        return embedings      
        
if __name__ == "__main__":
    batch = 1
    in_ch = 1
    H = W = D = 32
    nsteps = 4
    kernel_size = 3
    patch_size = 4
    embed_dim = 128
    
    model = MorphPatchEmb3D(in_ch, embed_dim, nsteps, patch_size, kernel_size).cuda()
    x = torch.randn(batch, in_ch, H, W, D).cuda()

    out = model(x)
    
    print("Input:", x.shape)
    print("Output:", out.shape)

