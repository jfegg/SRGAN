import torch
from torch import nn, Tensor
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class SSIM(nn.Module):
    """Implementation of SSIM in Pytorch"""


    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:

        print(raw_tensor.shape[0])

        # for i in range(raw_tensor.shape[0]):
        #     sr = raw_tensor[i, :, :, :]
        #     gt = dst_tensor[i, :, :, :]

        #     print(sr.shape)

        #     ssim_val = ssim(sr, gt, data_range=1)
        #     print(ssim_val)
        
        # exit()
        
        ssim_val = ssim(raw_tensor, dst_tensor, data_range=1, size_average=False, nonnegative_ssim=True)

        # for i in range(raw_tensor.shape[0]):
        #     for j in range(96):
        #         for k in range(96):
        #             print(raw_tensor[i, 0, j, k].item())

        # print(ssim_val)
        # print(ssim_val.size())

        ssim_val = torch.sub(1, ssim_val)

        # print(ssim_val)
        # print(ssim_val.size())

        # exit()
        
        return ssim_val