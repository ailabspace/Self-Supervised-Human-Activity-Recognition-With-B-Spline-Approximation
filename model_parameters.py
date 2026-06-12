from model.transformers import MPSCP

model = MPSCP(segments = [[2, 3], [2, 20], [4, 20], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22], [8, 20], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [1, 20], [0, 1], [0, 12], [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19]], d_ff = 1024, d_model = 256, num_heads= 8,
              num_layers_enc = 8, num_layers_dec=2, max_seq_length=8, dropout=0., t_m=15, body_avg=False)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
