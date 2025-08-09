import torch
checkpoint = torch.load("abc_normal/abc_normal", map_location='cpu')
print(checkpoint.keys())
