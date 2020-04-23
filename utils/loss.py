import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import log

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

class DiceCoefMultilabelLoss(nn.Module):

    def __init__(self, cuda=True):
        super().__init__()
        # self.smooth = torch.tensor(1., dtype=torch.float32)
        self.one = torch.tensor(1., dtype=torch.float32).cuda()
        self.activation = torch.nn.Softmax2d()

    def dice_loss(self, predict, target):
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = predict * target.cuda().float()
        score = (intersection.sum() * 2. + 1.) / (predict.sum() + target.sum() + 1.)
        return 1. - score

    def forward(self, predict, target, numLabels=4, channel='channel_first'):
        assert channel == 'channel_first' or channel == 'channel_last', r"channel has to be either 'channel_first' or 'channel_last'"
        dice = 0
        predict = self.activation(predict)
        if channel == 'channel_first':
            for index in range(numLabels):
                # Lme = [0.1, 0.1, 0.3, 0.5]
                temp = self.dice_loss(predict[:, index, :, :], target[:, index, :, :])
                dice += temp
        else:
            for index in range(numLabels):
                # Lme = [0.1, 0.1, 0.3, 0.5]
                temp = self.dice_loss(predict[:, :, :, index], target[:, :, :, index])
                dice += temp

        dice = dice / numLabels

        # entropy = entropy_loss(y_true) + ent
        return dice


class ShannonEntropyLoss(nn.Module):

    def __init__(self, cuda=True, loss_weight=1, n_class=4):
        super().__init__()
        self.loss_weight = loss_weight
        self.n_class = n_class
        self.activation = torch.nn.Softmax2d()

    def forward(self, predict):
        predict = self.activation(predict)
        return self.loss_weight * -(predict * torch.log(predict + 1e-20)).mean() / log(self.n_class)


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )





class JensenShannonDiv(nn.Module):
    """ Implementation of theJensenShannonDivergence for 2D images loss function from """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        from scipy.stats import entropy

    def forward(self, outputs, targets):
        output_pdf = Softmax_2D()(outputs)
        target_pdf = Softmax_2D()(targets)
        # normalize
        # output_pdf /= output_pdf.sum()
        # target_pdf /= target_pdf.sum()
        m = (output_pdf + target_pdf) / 2
        JSD = (self.entropy(output_pdf, m) + self.entropy(output_pdf, target_pdf)) / 2
        if self.reduction is 'mean':
            JSD = torch.mean(JSD)
        elif self.reduction is 'sum':
            JSD = torch.sum(JSD)
        return JSD

    @staticmethod
    def entropy(p, dim=-1, keepdim=None):
        # e = torch.distributions.Categorical(probs=p).entropy()
        return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

class KL_Divergence2D(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs, targets):
        p = Softmax_2D()(outputs)
        q = Softmax_2D()(targets)
        # D_KL(P || Q)

        # Reshae tensor to apply operation to 2D as: (batch * chanels * depth,  height * width)
        flat_dim = tuple(reduce(lambda x, y: x * y, outputs.shape[:-2]))
        spatial_dim = tuple(outputs.shape[-2] * outputs.shape[-1])
        kl = F.kl_div(
            q.view(flat_dim, spatial_dim).log(),
            p.view(flat_dim, spatial_dim),
            reduction='none',
        )
        kl_values = kl.sum(-1).view(outputs.shape[:-2])
        return kl_values


class AdaptiveWingLoss(nn.Module):
    """ implementation of the loss function from  Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression
      @InProceedings{Wang_2019_ICCV,
          author      = {Wang, Xinyao and Bo, Liefeng and Fuxin, Li},
          title       = {Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression},
          booktitle   = {The IEEE International Conference on Computer Vision (ICCV)},
          url         = {http://arxiv.org/abs/1904.07399},
          month       = {October},
          year        = {2019}
      }
      Github Repo: https://github.com/protossw512/AdaptiveWingLoss
      $$
      \begin{aligned}
      RWing(x) = \left\{ \begin{array}{ll}
                              0 &{} \text {if } |x|< r \\
                              w \ln (1 + (|x|-r)/\epsilon )   &{} \text {if } r \le |x| < w \\
                              |x| - C  &{} \text {otherwise}
                          \end{array}
                  \right. ,\nonumber \\
      \end{aligned}
      $$
      A = (1=(1 + (=)(􀀀y)))( 􀀀  y)((=)(􀀀y􀀀1))(1=)

      C = (A􀀀! ln(1+(=)􀀀y))
        """

    def __init__(self, reduction='mean', omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.alpha = float(alpha)
        self.w = float(omega)
        self.eps = float(epsilon)
        self.theta = float(theta)
        self.reduction = reduction

    def forward(self, outputs, targets):

        # outputs = Softmax_2D(outputs)
        # targets = Softmax_2D(targets)

        # Calculate thge smoothness factors
        A = self.w * (1 / (1 + (self.theta / self.eps) ** (self.alpha - targets))) * (self.alpha - targets) * (
                    (self.theta / self.eps) ** (self.alpha - targets - 1)) / self.eps
        C = self.theta * A - self.w * torch.log(1 + (self.theta / self.eps) ** (self.alpha - targets))

        # Select the case
        error_abs = torch.abs(outputs - targets)
        case1 = error_abs < self.theta
        case2 = error_abs >= self.theta

        # Compute the loss
        loss = torch.zeros_like(outputs)
        loss[case1] = self.w * torch.log( 1 + torch.abs((targets[case1] - outputs[case1]) / self.eps) ** (self.alpha - targets[case1]))
        loss[case2] = A[case2] * torch.abs(targets[case2] - outputs[case2]) - C[case2]

        if self.reduction is 'mean':
            return loss.mean()
        elif self.reduction is 'sum':
            return loss.sum()
        return loss

class RMSELoss(nn.Module):
    """ Calculate the RMSE loss for the 2 predicted heatmaps soft-landmark points
    :param outputs_coords: (torch.Tensor) outputs channelwise coordinats with the shape [Batch =8, channels=2, depth=32, xy=2]
    :param targets_coords: (torch.Tensor) targets heatmaps created from landmarks  shape [Batch =8, channels=2, depth=32,  xy=2]
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs_coords, targets_coords):
        # gt_targets_coords = torch.squeeze(gt_targets_coords, 1)  # Check the targets have dimension [B,Channels=2,D,H,W]
        l1 = torch.sqrt(torch.mean((outputs_coords[:, 0].float() - targets_coords[:, 0].float()) ** 2))  # [C=0,32,2]
        l2 = torch.sqrt(torch.mean((outputs_coords[:, 1].float() - targets_coords[:, 1].float()) ** 2))  # [C=0,32,2]
        if self.reduction is 'sum':
            loss = torch.sum(torch.stack([l1, l2]).float())
        else:
            loss = torch.mean(torch.stack([l1, l2]).float())


        return loss
