import torch
from torch import nn, Tensor
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class SSIM(nn.Module):
    """Implementation of SSIM in Pytorch"""


    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        
        ssim_val = ssim(raw_tensor, dst_tensor, data_range=255, nonnegative_ssim=True)
        ssim_val = torch.mul(ssim_val, -1)
        ssim_val = torch.add(ssim_val, 1)
        
        return ssim_val