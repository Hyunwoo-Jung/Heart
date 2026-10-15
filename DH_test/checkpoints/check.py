import torch

ckpt = torch.load(
    "/home/dandong00/Projects/Heart/DH_test/checkpoints/best.pt",
    map_location="cpu",
    weights_only=True,
)
print(ckpt["evaluation"])