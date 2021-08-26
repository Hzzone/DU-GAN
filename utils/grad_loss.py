import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4)

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.sobel = SobelOperator(1e-4)

    def forward(self, pr, gt):
        gt_sobel = self.sobel(gt)
        pr_sobel = self.sobel(pr)
        grad_loss = F.l1_loss(gt_sobel, pr_sobel)
        return grad_loss
