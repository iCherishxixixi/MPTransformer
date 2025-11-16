import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DfMap(nn.Module):
    def __init__(self, in_ch, nsteps, kernel_size):
        super(DfMap, self).__init__()
        self.vec_conv = nn.Conv2d(in_ch, 2 * in_ch, kernel_size, padding=(kernel_size - 1) // 2)        
        self.bn = nn.BatchNorm2d(2 * in_ch)
        
        self.VecInt = VecInt(nsteps)
        self.transformer = SpatialTransformer()
        
        self.fuse_conv = nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=(nsteps, 1, 1))

    def forward(self, f):
        device = f.device
        
        vec = self.vec_conv(f)
        vec = self.bn(vec)
        # offset = torch.tanh(offset)
        bs, c, h, w = vec.shape
        vec = vec.view(bs*c//2, 2, h, w)
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in [h, w]]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).requires_grad_(False).to(device)
        
        deforms = self.VecInt(vec, grid)
        finalmaps = []
        
        for j in range(len(deforms)):
            finalmap = self.transformer(f.view(bs*c//2, 1, h, w), deforms[j], grid)
            finalmap = finalmap.view(bs, c//2, h, w)
            finalmaps.append(finalmap)
        # stack: (nsteps, B, C, H, W) -> (B, C, nsteps, H, W)
        stacked = torch.stack(finalmaps, dim=0).permute(1, 2, 0, 3, 4)
        fused = self.fuse_conv(stacked).squeeze(2)
        return fused
  

    
    