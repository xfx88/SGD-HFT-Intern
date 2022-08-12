import torch
from torch import nn

class WeightedMSE(nn.Module):
    def __init__(self, device, scale_factor = 1.):
        """
        :param device: 在DDP模式中需要传入当前子进程对应的local_rank
        """
        super().__init__()
        self.scale_factor = scale_factor
        # 此处指定各输出的权重
        self.weight = torch.Tensor([0.1, 0.2, 0.3, 0.4]).to(device)
        self.base_loss = nn.MSELoss(reduction = "none")

    def forward(self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor) -> torch.Tensor:

        loss = torch.mean(self.scale_factor * (self.base_loss(y_pred, y_true) * self.weight), dim = 0)
        return loss.mean()


if __name__ == "__main__":
    criterion = WeightedMSE(device = 0)
    pred = torch.randn((4,4)).to(0)
    y_true = torch.randn((4,4)).to(0)
    loss = criterion(y_true, pred)