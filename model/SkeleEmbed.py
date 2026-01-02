import torch
import torch.nn as nn

class SpatialEmbedderJustCNN(nn.Module):
    def __init__(self, in_channels, out_channels, j_patch_size, t_patch_size):
        super(SpatialEmbedderJustCNN, self).__init__()

        self.t_patch_size = t_patch_size

        self.out_channels = out_channels

        self.kernel = (1, t_patch_size, j_patch_size)

        # N, S, F, J, C
        self.coord_projector =  nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel, stride=self.kernel)

    def forward(self, x):

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.coord_projector(x).permute(0, 2, 3, 4, 1).contiguous()

        return x

class SpatialEmbedderCNN(nn.Module):
    def __init__(self, in_channels, out_channels, j_patch_size, t_patch_size):
        super(SpatialEmbedderCNN, self).__init__()

        self.t_patch_size = t_patch_size

        self.out_channels = out_channels

        self.kernel_1 = (1, t_patch_size, 1)

        self.kernel_2 = (1, 1, j_patch_size)

        # N, S, F, J, C
        self.coord_projector =  nn.Conv3d(in_channels, out_channels // 2, kernel_size=self.kernel_1, stride=self.kernel_1)

        self.joint_rel_projector = nn.Conv3d(out_channels // 2, out_channels, kernel_size=self.kernel_2, stride=self.kernel_2)

        #self.norm1 = nn.LayerNorm(out_channels//2)

        #self.norm2 = nn.LayerNorm(out_channels)

        self.act_1 = nn.ReLU()

        self.act_2 = nn.ReLU()


    def forward(self, x):
        N, S, F, J, C = x.shape

        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x = self.act_1(self.coord_projector(x).permute(0, 2, 3, 4, 1).contiguous()).permute(0, 4, 1, 2, 3).contiguous()

        x = self.act_2(self.joint_rel_projector(x).permute(0, 2, 3, 4, 1).contiguous())

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, S, F//self.t_patch_size, self.out_channels)

        return x

class SCNNEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels, j_patch_size, t_patch_size):
        super(SCNNEmbedder, self).__init__()

        self.t_patch_size = t_patch_size

        self.out_channels = out_channels

        self.kernel = (1, t_patch_size)

        # N, S, F, J, C
        self.coord_projector =  nn.Conv2d(in_channels*j_patch_size, out_channels, kernel_size=self.kernel, stride=self.kernel)


    def forward(self, x):
        N, S, F, J, C = x.shape

        x = x.view(N, S, F, -1)

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.coord_projector(x).permute(0, 2, 3, 1).contiguous()

        return x

class SpatialEmbedderLinear(nn.Module):
    def __init__(self, in_channels, out_channels, j_patch_size, t_patch_size):
        super(SpatialEmbedderLinear, self).__init__()

        self.t_patch_size = t_patch_size

        self.out_channels = out_channels

        #self.linear_coords_projector = nn.Linear(t_patch_size*in_channels, 2**t_patch_size)
        self.linear_coords_projector = nn.Linear(t_patch_size*in_channels, out_channels//4)

        #self.linear_joints_projector = nn.Linear(j_patch_size*2**t_patch_size, out_channels)
        self.linear_joints_projector = nn.Linear(j_patch_size*out_channels//4, out_channels)

        self.act_1 = nn.GELU()

        self.act_2 = nn.GELU()

        #self.norm_1 = nn.LayerNorm(2**t_patch_size)

        #self.norm_2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        N, S, F, J, C = x.shape

        x = x.permute(0, 1, 3, 2, 4).contiguous().view(N, S, J, -1, self.t_patch_size*C)

        x = self.act_1(self.linear_coords_projector(x))
        #x = self.act_1(self.norm_1(self.linear_coords_projector(x)))

        W = x.shape[3]

        x = x.permute(0, 1, 3, 2, 4).contiguous().view(N, S, W, -1)

        x = self.act_2(self.linear_joints_projector(x))
        #x = self.act_2(self.norm_2(self.linear_joints_projector(x)))

        return x

class SplineEmbedderLinear(nn.Module):
    def __init__(self, in_channels, out_channels, j_patch_size, t_patch_size):
        super(SplineEmbedderLinear, self).__init__()

        self.t_patch_size = t_patch_size

        self.out_channels = out_channels

        #self.linear_coords_projector = nn.Linear(t_patch_size*in_channels, 2**t_patch_size)
        self.linear_coords_projector = nn.Linear(t_patch_size*in_channels, out_channels//4)

        #self.linear_joints_projector = nn.Linear(j_patch_size*2**t_patch_size, out_channels)
        self.linear_joints_projector = nn.Linear(j_patch_size*out_channels//4, out_channels)

        self.act_1 = nn.GELU()

        self.act_2 = nn.GELU()

        #self.norm_1 = nn.LayerNorm(2**t_patch_size)

        #self.norm_2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        N, S, F, J, C = x.shape

        x = x.permute(0, 1, 3, 2, 4).contiguous().view(N, S, J, -1, self.t_patch_size*C)

        x = self.act_1(self.linear_coords_projector(x))
        #x = self.act_1(self.norm_1(self.linear_coords_projector(x)))

        W = x.shape[3]

        x = x.permute(0, 1, 3, 2, 4).contiguous().view(N, S, W, -1)

        x = self.act_2(self.linear_joints_projector(x))
        #x = self.act_2(self.norm_2(self.linear_joints_projector(x)))

        return x

class DeepSetLinear(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim = 32, j_patch_size = 6, t_patch_size = 4):
        super(DeepSetLinear, self).__init__()

        self.t_patch_size = t_patch_size

        self.out_channels = out_channels

        #self.linear_coords_projector = nn.Linear(t_patch_size*in_channels, 2**t_patch_size)
        self.phi = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_channels),
            nn.GELU()
        )


    def forward(self, x):
        N, S, F, J, C = x.shape

        x = x.view(N, S, -1, 4, J, C)

        phi_x = self.phi(x)

        pooled = phi_x.mean(dim = (-3, -2))

        return self.rho(pooled)

class DeepSplineLinear(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim = 32):
        super(DeepSplineLinear, self).__init__()

        self.out_channels = out_channels

        self.phi = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_channels),
            nn.GELU()
        )


    def forward(self, x):
        N, S, W, J, C, coeffs = x.shape

        phi_x = self.phi(x)

        pooled = phi_x.mean(dim = (-3, -2))

        return self.rho(pooled)
