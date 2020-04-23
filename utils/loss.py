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
