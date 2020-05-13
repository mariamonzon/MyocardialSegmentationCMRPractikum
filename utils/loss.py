#!/usr/bin/env python3.7
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import log
from torch import Tensor,  einsum
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from typing import  Iterable, Set
import numpy as np

# Helper Functions
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


class LossMeter:
    def __init__(self):
        self.loss_value = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value,  n=1):
        """Update the counters parameters for the Loss"""
        self.sum +=  value * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg_loss(self):
        return self.avg


class DiceCoefMultilabelLoss(nn.Module):

    def __init__(self, numLabels=4 ):
        super().__init__( )
        self.activation = torch.nn.Softmax2d()
        self.numLabels = numLabels

    def forward(self, predict, target):
        dice = 0
        predict = self.activation(predict)

        for c in range(self.numLabels):
            # Lme = [0.1, 0.1, 0.3, 0.5]
            dice += self.dice_coeff(predict[:, c, :, :], target[:, c, :, :])
        dice = dice /self.numLabels

        return dice

    @staticmethod
    def dice_coeff(predict, target, smooth=1.):
        intersection = predict.contiguous().view(-1) *  target.contiguous().view(-1)
        score = (intersection.sum() * 2. + smooth) / (predict.sum() + target.sum()  + smooth)
        return 1. - score
        # intersection = einsum("bcwh,bcwh->bc", predict, target)
        # union = (einsum("bcwh->bc", predict) + einsum("bcwh->bc", target))
        # loss =  1 - (2 * intersection + 1e-10) / (union + 1e-10)


class SurfaceLoss(nn.Module):

    def __init__(self, numLabels=4):
        super().__init__()
        self.activation = torch.nn.Softmax2d()
        self.numLabels = numLabels

    def forward(self, predict, target_maps):
    # def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        loss = 0.
        for c in range(self.numLabels):
            loss += predict[:, c, :, :].mul(target_maps[:, c, :, :])
        return loss/self.numLabels

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    @staticmethod
    def one_hot(label, n_classes, requires_grad=True):
        """Return One Hot Label"""
        one_hot_label = torch.eye(
            n_classes, device=label.device, requires_grad=requires_grad)[label]
        one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)
        return one_hot_label

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)
        return loss


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


class TverskyLoss(nn.Module):
    def __init__(self, alpha = 0.5 ,  beta = 0.5):
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true, weight=None):
        ones = torch.ones_like(y_true)
        p0 = y_pred
        p1 = ones - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = ones - y_true

        num = torch.sum(p0 * g0, (0, 1, 2, 3))
        den = num + self.alpha * torch.sum(p0 * g1, (0, 1, 2, 3)) + self.beta * torch.sum(p1 * g0, (0, 1, 2, 3))

        T = torch.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

        Ncl = torch.cast(torch.shape(y_true)[-1], 'float32')
        return Ncl - T


        def focal_tversky(self, y_true, y_pred):
            pt_1 = self.forward(self, y_pred, y_true, weight=None)
            gamma = 0.75
            return torch.pow((1 - pt_1), gamma)




class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, classes=4, sigmoid_normalization=True, skip_index_after=None, epsilon=1e-6,):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.classes = None
        self.skip_index_after = None
        #~        self.register_buffer('weight', weight)
        self.epsilon = epsilon
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    @staticmethod
    def expand_as_one_hot(input, C, ignore_index=None):
        """
        Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :param ignore_index: ignore index to be kept during the expansion
        :return: 5D output image (NxCxDxHxW)
        """
        if input.dim() == 5:
            return input
        assert input.dim() == 4

        # expand the input tensor to Nx1xDxHxW before scattering
        input = input.unsqueeze(1)
        # create result tensor shape (NxCxDxHxW)
        shape = list(input.size())
        shape[1] = C

        if ignore_index is not None:
            # create ignore_index mask for the result
            mask = input.expand(shape) == ignore_index
            # clone the lib tensor and zero out ignore_index in the input
            input = input.clone()
            input[input == ignore_index] = 0
            # scatter to get the one-hot tensor
            result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
            # bring back the ignore_index in the result
            result[mask] = ignore_index
            return result
        else:
            # scatter to get the one-hot tensor
            return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


    def compute_per_channel_dice(self,input, target, epsilon=1e-6, weight=None):
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
        Args:
             input (torch.Tensor): NxCxSpatial input tensor
             target (torch.Tensor): NxCxSpatial target tensor
             epsilon (float): prevents division by zero
             weight (torch.Tensor): Cx1 tensor of weight per channel/class
        """

        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input  = self.flatten(input)
        target = self.flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        if weight is not None:
            intersect = weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        return 2 * (intersect / denominator.clamp(min=epsilon))

    @staticmethod
    def flatten(tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
        """
        # number of channels
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, H, W) -> (C, N, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, H, W) -> (C, N * H * W)
        return transposed.contiguous().view(C, -1)

    def dice(self, input, target, weight):
        assert input.size() == target.size()
        input  = self.flatten(input)
        target = self.flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

    def forward(self, input, target):
        """
        Expand to one hot added extra for consistency reasons
        """
        target = self.expand_as_one_hot(target.long(), self.classes)

        assert input.dim() == target.dim() == 5 ,"'input' and 'target' have different number of dims"

        if self.skip_index_after is not None:
            target = self.skip_target_channels(target, self.skip_index_after)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)
        loss = (1. - torch.mean(per_channel_dice))
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        # average Dice score across all channels/classes
        return loss, per_channel_dice


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
        self.norm =  nn.Softmax2d()
        from scipy.stats import entropy

    def forward(self, outputs, targets):
        # normalize to be a pdf
        output_pdf = self.norm(outputs)
        target_pdf = self.norm(targets)
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
        p = nn.Softmax2d()(outputs)
        q = nn.Softmax2d()(targets)
        # D_KL(P || Q)

        # Reshae tensor to apply operation to 2D as: (batch * chanels * depth,  height * width)
        flat_dim = np.prod(outputs.shape[:-2])
        spatial_dim = tuple(outputs.shape[-2] * outputs.shape[-1])
        kl = F.kl_div(
            q.view(flat_dim, spatial_dim).log(),
            p.view(flat_dim, spatial_dim),
            reduction='none',
        )
        kl_values = kl.sum(-1).view(outputs.shape[:-2])
        return kl_values