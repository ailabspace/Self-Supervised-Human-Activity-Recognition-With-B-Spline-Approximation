import torch.nn as nn

class CNNProjector(nn.Module):
    def __init__(self, in_channels, out_channels, j_patch_size, t_patch_size):
        super(CNNProjector, self).__init__()

        self.t_patch_size = t_patch_size

        self.out_channels = out_channels

        self.kernel = (1, t_patch_size, j_patch_size)

        # N, S, F, J, C
        self.coord_projector =  nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel, stride=self.kernel)

    def forward(self, x):

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.coord_projector(x).permute(0, 2, 3, 4, 1).contiguous()

        return x
