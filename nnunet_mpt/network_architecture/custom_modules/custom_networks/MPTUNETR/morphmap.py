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
        self.vec_conv = nn.Conv3d(in_ch, 3 * in_ch, kernel_size, padding=(kernel_size - 1) // 2)        
        self.bn = nn.BatchNorm3d(3 * in_ch)
        
        self.VecInt = VecInt(nsteps)
        self.transformer = SpatialTransformer()

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
        return finalmaps
  
class DfWin(nn.Module):
    def __init__(self, in_ch, kernel_size, nsteps=7):
        super(DfWin, self).__init__()
        self.vec_conv = nn.Conv3d(in_ch, 3, kernel_size, padding=(kernel_size - 1) // 2)        
        self.bn = nn.BatchNorm3d(3)
        
        self.VecInt = VecInt(nsteps)
        self.transformer = SpatialTransformer()

    def forward(self, f):
        device = f.device
        f = f.permute(0, 4, 1, 2, 3)
        
        vec = self.vec_conv(f)
        vec = self.bn(vec)
        # offset = torch.tanh(offset)
        bs, c, l, h, w = vec.shape
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in [l, h, w]]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).requires_grad_(False).to(device)
        
        deforms = self.VecInt(vec, grid)
        finalmap = self.transformer(f, deforms[-1], grid)
        finalmap = finalmap.permute(0, 2, 3, 4, 1)
        return finalmap
        
class DfWin2(nn.Module):
    def __init__(self, in_ch, kernel_size, scope=3, mode='bilinear'):
        super(DfWin2, self).__init__()
        self.df_conv = nn.Conv3d(in_ch, 3, kernel_size, padding=(kernel_size - 1) // 2)        
        self.bn = nn.BatchNorm3d(3)
        self.mode = mode
        self.scope = scope

    def forward(self, f):
        device = f.device
        f = f.permute(0, 4, 1, 2, 3)
        
        offset = self.df_conv(f)
        offset = self.bn(offset)*self.scope
        offset = torch.tanh(offset)
        bs, c, l, h, w = f.shape
        
        # create sampling grid
        vectors = [torch.linspace(-1, 1, s) for s in [l, h, w]]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).repeat(bs, 1, 1, 1, 1)
        grid = grid.type(torch.FloatTensor).requires_grad_(False).to(device)
        
        grid = torch.stack([offset[:, 0,...]/l, offset[:, 1,...]/h, offset[:, 2,...]/w], dim=1)
         
        new_grid = grid + offset
        new_grid = torch.clamp(new_grid, min=-1, max=1).permute(0, 2, 3, 4, 1)
        
        dfmap = F.grid_sample(f, new_grid, align_corners=True, mode=self.mode)
        dfmap = dfmap.permute(0, 2, 3, 4, 1)
        return dfmap
        
        
if __name__ == "__main__":
    input_tensor = torch.randn(2, 128, 128, 128, 48) 
    model = DfWin2(48, 3)
    output = model(input_tensor)

    print("Output shape:", output.shape)
    
    