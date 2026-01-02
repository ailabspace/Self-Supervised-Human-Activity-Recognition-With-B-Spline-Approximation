import math

import torch
import torch.nn as nn

import numpy as np


def compute_bspline_coefficients(x):
    """ frames: [4, d] """
    # B = torch.tensor([
    #     [-1 / 6, 3 / 6, -3 / 6, 1 / 6],
    #     [3 / 6, -6 / 6, 3 / 6, 0 / 6],
    #     [-3 / 6, 0 / 6, 3 / 6, 0 / 6],
    #     [1 / 6, 4 / 6, 1 / 6, 0 / 6]
    # ], device=x.device, dtype=torch.float32)  # Basis matrix

    B = torch.tensor([
        [1 / 6, 4 / 6, 1 / 6, 0 / 6],
        [-3 / 6, 0 / 6, 3 / 6, 0 / 6],
        [3 / 6, -6 / 6, 3 / 6, 0 / 6],
        [-1 / 6, 3 / 6, -3 / 6, 1 / 6]
    ], device=x.device, dtype=torch.float32)  # Basis matrix

    N, F, J = x.shape

    x = x.permute(0, 2, 1).reshape(-1, F)

    return torch.mm(B, x.T).view(-1, N, J).permute(1, 2, 0)


def compute_splines(x, t_m):
    N, F, J, C = x.shape
    W = int(F / t_m)

    x, y, z = x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]

    splines = torch.zeros(N, W, J, C, t_m - 3, 4, device = x.device)

    for w in range(W):
        for t in range(0, t_m - 3):
            f = w * t_m + t
            splines[:, w, :, 0, t] = compute_bspline_coefficients(x[:, f:f + 4])
            splines[:, w, :, 1, t] = compute_bspline_coefficients(y[:, f:f + 4])
            splines[:, w, :, 2, t] = compute_bspline_coefficients(z[:, f:f + 4])

    # print("X Coordinates: ", x[0, 0:4, 0])
    # print("Spline: ", splines[0, 0, 0, 0])
    return splines.view(N, W, J, C, -1)


def reconstruct_coords(s, t):
    """
    :param s: spline with shape N, E, J, C, 4
    :param t: values of t for splines N, E, J, C, 4
    :return: x
    """

    # s [N, E, J, C, 4] -> [N, E, J, C, 4, 4]
    t = torch.stack([
        torch.ones_like(t, device = t.device),
        t,
        t**2,
        t**3
    ], dim =-1)

    t = t.unsqueeze(-2)

    #print(t.shape)

    N, E, J, C, T, _, _ = t.shape

    #print(t[0, 0, 0, 0, 0])

    s = s.unsqueeze(-2).unsqueeze(-1).repeat(1, 1, 1, 1, T, 1, 1)

    #print(s.shape)

    #print(s[0,0,0,0,0])

    x = torch.matmul(t, s)

    #print(x[0,0,0,0,0])

    return x.squeeze(-1).squeeze(-1)

def compute_displacement(x):
    if x.dim() == 5:
        N, S, F, J, C = x.shape
        x_motion = torch.zeros_like(x)
        x_motion[:, :, :-1] = x[:, :, :-1] - x[:, :, 1:]
        x_motion[:, :, -1] = 0
    else:
        N, F, J, C = x.shape
        x_motion = torch.zeros_like(x)
        x_motion[:, :-1] = x[:, :-1] - x[:, 1:]
        x_motion[:, -1] = 0

    return x_motion

def compute_accelleration(x):
    x_vel = compute_displacement(x)
    return compute_displacement(x_vel)

def compute_bones(x, bones):
    N, F, J, C = x.shape
    x_bones = torch.zeros(N, F, len(bones), C, device = x.device)
    for i, (v1, v2)in enumerate(bones):
        x_bones[:, :, i] = x[:, :, v2] - x[:, :, v1]

    return x_bones

def find_bone_segments(segments, bones):
    bone_segments = []
    for segment in segments:
        s = set(segment)
        matches = [i for i, pair in enumerate(bones) if pair[0] in s and pair[1] in s]
        bone_segments.append(matches)

    return bone_segments

# def compute_displacement_segments(x, t_m = 4):
#     N, S, F, J, C = x.shape
#     W = F//t_m
#     x_motion = torch.zeros(N, S, W, t_m - 3, J, C)
#     for w in range(W):
#         for t in range(0, t_m - 3):
#             f = w * t_m + t
#             motion_t = torch.zeros_like()
#             x_motion[:, :, w, t] = x[:, :, f]
#     return x_motion

def compute_kinetic_energy(x, t_m, per_joints = False, windowed = True):
    N, F, J, C = x.shape
    W = F//t_m

    x_motion = torch.zeros_like(x)
    x_motion[:, :-1] = x[:, :-1] - x[:, 1:]
    x_motion[:, -1] = 0

    if per_joints:
        x_energy = torch.sum(x_motion**2, dim = [-1])
        if windowed:
            x_energy = x_energy.view(N, W, t_m, J)
    else:
        x_energy = torch.sum(x_motion**2, dim = [-1, -2])
        if windowed:
            x_energy = x_energy.view(N, W, t_m)

    return x_energy

def compute_kinetic_splines(x, t_m):
    N, F, J = x.shape
    W = int(F / t_m)

    splines = torch.zeros(N, W, J, t_m - 3, 4, device = x.device)

    for w in range(W):
        for t in range(0, t_m - 3):
            f = w * t_m + t
            splines[:, w, :, t] = compute_bspline_coefficients(x[:, f:f + 4])

    return splines.view(N, W, J, -1)

def calculate_kinetic_energy(x):
    N, S, W, F, J, C = x.shape

    x = x.view(N, -1, F, J, C)

    x_motion = torch.zeros_like(x)
    x_motion[:, :, :-1] = x[:, :, :-1] - x[:, :, 1:]
    x_motion[:, :, -1] = x_motion[:, :, -2]

    x_motion = 0.5*(x_motion**2)
    x_motion = x_motion.sum(dim = (-1, -2, -3))

    return x_motion.view(N, S*W, -1)

def generate_mask_frames(sequence, frame_mask_ratio):
    x = sequence.clone()

    N, F, C = x.shape

    masked_frames = int(F*frame_mask_ratio)
    if masked_frames < 1:
        masked_frames = 1

    random = torch.rand(N, F, device = x.device)
    idx_shuffle = torch.argsort(random, dim = 1)
    idx_restore = torch.argsort(idx_shuffle, dim = 1)

    idx_predict = idx_shuffle[:, :masked_frames]
    idx_kept = idx_shuffle[:, masked_frames:]

    x = torch.gather(x, dim=1, index=idx_kept.unsqueeze(-1).repeat(1, 1, C))

    return x, idx_predict, idx_kept, idx_restore

def kinetic_mask_frames(sequence, sequence_original, frame_mask_ratio):
    x = sequence.clone()

    N, F, C = x.shape

    masked_frames = int(F*frame_mask_ratio)
    if masked_frames < 1:
        masked_frames = 1

    hi_samples = int(N * 0.5)
    lo_samples = int(N * 0.4)

    energy = calculate_kinetic_energy(sequence_original)

    energy = energy.squeeze(-1)

    idx_shuffle_hi = torch.argsort(energy, dim = 1, descending=True)
    idx_restore_hi = torch.argsort(idx_shuffle_hi, dim = 1)
    idx_restore_hi = idx_restore_hi[:hi_samples]

    idx_predict_hi = idx_shuffle_hi[:hi_samples, :masked_frames]
    idx_kept_hi = idx_shuffle_hi[:hi_samples, masked_frames:]

    idx_shuffle_lo = torch.argsort(energy, dim = 1)
    idx_restore_lo = torch.argsort(idx_shuffle_lo, dim = 1)
    idx_restore_lo = idx_restore_lo[hi_samples:hi_samples+lo_samples]

    idx_predict_lo = idx_shuffle_lo[hi_samples:hi_samples+lo_samples, :masked_frames]
    idx_kept_lo = idx_shuffle_lo[hi_samples:hi_samples+lo_samples, masked_frames:]

    random = torch.rand(N, F, device=x.device)
    idx_shuffle_rand = torch.argsort(random, dim=1)
    idx_restore_rand = torch.argsort(idx_shuffle_rand, dim=1)
    idx_restore_rand = idx_restore_rand[hi_samples+lo_samples:]

    idx_predict_rand = idx_shuffle_rand[hi_samples+lo_samples:, :masked_frames]
    idx_kept_rand = idx_shuffle_rand[hi_samples+lo_samples:, masked_frames:]

    idx_predict = torch.cat([idx_predict_hi, idx_predict_lo, idx_predict_rand], dim = 0)
    idx_kept = torch.cat([idx_kept_hi, idx_kept_lo, idx_kept_rand], dim = 0)
    idx_restore = torch.cat([idx_restore_hi, idx_restore_lo, idx_restore_rand], dim = 0)

    x = torch.gather(x, dim=1, index=idx_kept.unsqueeze(-1).repeat(1, 1, C))

    return x, idx_predict, idx_kept, idx_restore

def add_mask_tokens(x, idx_restore, mask_token_frame):
    N, F, C = x.shape
    mask_tokens = mask_token_frame.repeat(N, idx_restore.shape[1] - x.shape[1], 1)

    x = torch.cat([x, mask_tokens], dim = 1)
    x = torch.gather(x, dim =1, index = idx_restore.unsqueeze(-1).repeat(1, 1, C))

    return x

def generate_mask_frames2d(sequence, frame_mask_ratio):
    x = sequence.clone()
    N, S, F, C = x.shape

    masked_segments = int(S*frame_mask_ratio)
    if masked_segments < 1:
        masked_segments = 1

    masked_frames = int(F*frame_mask_ratio)
    if masked_frames < 1:
        masked_frames = 1

    # MASK SEGMENTS
    random_s = torch.rand(N, S, device=x.device)
    idx_shuffle_s = torch.argsort(random_s, dim=1)
    idx_restore_s = torch.argsort(idx_shuffle_s, dim=1)

    idx_predict_s = idx_shuffle_s[:, :masked_segments]
    idx_kept_s = idx_shuffle_s[:, masked_segments:]

    x = torch.gather(x, dim = 1, index = idx_kept_s.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, C))

    # MASK WINDOWS
    random_f = torch.rand(N, F, device = x.device)
    idx_shuffle_f = torch.argsort(random_f, dim = 1)
    idx_restore_f = torch.argsort(idx_shuffle_f, dim = 1)

    idx_predict_f = idx_shuffle_f[:, :masked_frames]
    idx_kept_f = idx_shuffle_f[:, masked_frames:]

    x = torch.gather(x, dim=2, index=idx_kept_f.unsqueeze(1).unsqueeze(-1).repeat(1, idx_kept_s.shape[-1], 1, C))

    return x, idx_predict_s, idx_predict_f, idx_kept_s, idx_kept_f, idx_restore_s, idx_restore_f

def add_mask_tokens2d(x, idx_restore_s, idx_restore_f, mask_token):
    N, S, F, C = x.shape
    mask_tokens_s = mask_token.unsqueeze(1).repeat(N, idx_restore_s.shape[1] - S, idx_restore_f.shape[1], 1)
    mask_tokens_f = mask_token.unsqueeze(1).repeat(N, S, idx_restore_f.shape[1] - F, 1)

    x = torch.cat([x, mask_tokens_f], dim = 2)
    x = torch.gather(x, dim = 2, index = idx_restore_f.unsqueeze(1).unsqueeze(-1).repeat(1, S, 1, C))

    x = torch.cat([x, mask_tokens_s], dim = 1)
    x = torch.gather(x, dim =1, index = idx_restore_s.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, idx_restore_f.shape[1], C))

    return x

def generate_mask_windows(sequence, frame_mask_ratio):
    x = sequence.clone()
    N, S, F, C = x.shape

    masked_frames = int(F*frame_mask_ratio)
    if masked_frames < 1:
        masked_frames = 1

    # MASK WINDOWS
    random_f = torch.rand(N, F, device = x.device)
    idx_shuffle_f = torch.argsort(random_f, dim = 1)
    idx_restore_f = torch.argsort(idx_shuffle_f, dim = 1)

    idx_predict_f = idx_shuffle_f[:, :masked_frames]
    idx_kept_f = idx_shuffle_f[:, masked_frames:]

    x = torch.gather(x, dim=2, index=idx_kept_f.unsqueeze(1).unsqueeze(-1).repeat(1, S, 1, C))

    return x, idx_predict_f, idx_kept_f, idx_restore_f

def add_mask_windows(x, idx_restore_f, mask_token):
    N, S, F, C = x.shape
    mask_tokens_f = mask_token.unsqueeze(1).repeat(N, S, idx_restore_f.shape[1] - F, 1)

    x = torch.cat([x, mask_tokens_f], dim = 2)
    x = torch.gather(x, dim = 2, index = idx_restore_f.unsqueeze(1).unsqueeze(-1).repeat(1, S, 1, C))

    return x

def generate_mask_segments(sequence, frame_mask_ratio):
    x = sequence.clone()
    N, S, F, C = x.shape

    masked_segments = int(S*frame_mask_ratio)
    if masked_segments < 1:
        masked_segments = 1

    # MASK SEGMENTS
    random_s = torch.rand(N, S, device=x.device)
    idx_shuffle_s = torch.argsort(random_s, dim=1)
    idx_restore_s = torch.argsort(idx_shuffle_s, dim=1)

    idx_predict_s = idx_shuffle_s[:, :masked_segments]
    idx_kept_s = idx_shuffle_s[:, masked_segments:]

    x = torch.gather(x, dim = 1, index = idx_kept_s.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, C))

    return x, idx_predict_s, idx_kept_s, idx_restore_s

def add_mask_segments(x, idx_restore_s, mask_token):
    N, S, F, C = x.shape
    mask_tokens_s = mask_token.unsqueeze(1).repeat(N, idx_restore_s.shape[1] - S, F, 1)

    x = torch.cat([x, mask_tokens_s], dim = 1)
    x = torch.gather(x, dim =1, index = idx_restore_s.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, C))

    return x

def compute_loss(recon, truth, norm=False, idx_predict=None, loss_fn = nn.MSELoss()):
    if norm:
        mean = truth.mean(dim=-1, keepdim=True)
        var = truth.var(dim=-1, keepdim=True)
        truth = (truth - mean) / (var + 1.0e-6) ** 0.5

    # Ground truth: [N, W, t_m, J, C] -> [N, frame_mask_ratio*W, t_m, J, C]
    target_tp = torch.gather(truth, dim=1,
                             index=idx_predict.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, truth.shape[-3], truth.shape[-2], truth.shape[-1]))

    return loss_fn(recon, target_tp)

def compute_loss_3(recon, truth, norm=False, idx_predict=None, loss_fn = nn.MSELoss()):
    if norm:
        mean = truth.mean(dim=-1, keepdim=True)
        var = truth.var(dim=-1, keepdim=True)
        truth = (truth - mean) / (var + 1.0e-6) ** 0.5

    # Ground truth: [N, W, t_m, J, C] -> [N, frame_mask_ratio*W, t_m, J, C]
    target_tp = torch.gather(truth, dim=1,
                             index=idx_predict.unsqueeze(-1).repeat(1, 1, truth.shape[-1]))

    return loss_fn(recon, target_tp)

def compute_loss_4(recon, truth, norm=False, idx_predict=None, loss_fn = nn.MSELoss()):
    if norm:
        mean = truth.mean(dim=-1, keepdim=True)
        var = truth.var(dim=-1, keepdim=True)
        truth = (truth - mean) / (var + 1.0e-6) ** 0.5

    # Ground truth: [N, W, t_m, J, C] -> [N, frame_mask_ratio*W, t_m, J, C]
    target_tp = torch.gather(truth, dim=1,
                             index=idx_predict.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, truth.shape[-2], truth.shape[-1]))

    return loss_fn(recon, target_tp)

def compute_loss2d(recon, truth, norm=False, idx_predict_s = None, idx_predict_f = None, loss_fn = nn.MSELoss()):
    if norm:
        mean = truth.mean(dim=-1, keepdim=True)
        var = truth.var(dim=-1, keepdim=True)
        truth = (truth - mean) / (var + 1.0e-6) ** 0.5

    target_tp = torch.gather(truth, dim=1,
                             index=idx_predict_s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, truth.shape[-4], truth.shape[-3], truth.shape[-2], truth.shape[-1]))

    target_tp = torch.gather(target_tp, dim = 2,
                             index = idx_predict_f.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, target_tp.shape[1], 1, truth.shape[-3], truth.shape[-2], truth.shape[-1]))

    return loss_fn(recon, target_tp)

def compute_loss_window(recon, truth, norm=False, idx_predict_f = None, loss_fn = nn.MSELoss()):
    if norm:
        mean = truth.mean(dim=-1, keepdim=True)
        var = truth.var(dim=-1, keepdim=True)
        truth = (truth - mean) / (var + 1.0e-6) ** 0.5

    target_tp = torch.gather(truth, dim = 2,
                             index = idx_predict_f.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, truth.shape[1], 1, truth.shape[-3], truth.shape[-2], truth.shape[-1]))

    return loss_fn(recon, target_tp)

def compute_loss_segments(recon, truth, norm=False, idx_predict_s = None, loss_fn = nn.MSELoss()):
    if norm:
        mean = truth.mean(dim=-1, keepdim=True)
        var = truth.var(dim=-1, keepdim=True)
        truth = (truth - mean) / (var + 1.0e-6) ** 0.5

    target_tp = torch.gather(truth, dim=1,
                             index=idx_predict_s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, truth.shape[-4], truth.shape[-3], truth.shape[-2], truth.shape[-1]))

    return loss_fn(recon, target_tp)

if __name__ == '__main__':
    x = torch.rand(255, 5, 6, 3, 4).cuda()
    coeff = torch.rand(255, 5, 6, 3, 4).cuda()
    reconstruct_coords(coeff, x)