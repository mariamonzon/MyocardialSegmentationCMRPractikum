import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import log

torch.set_default_tensor_type('torch.cuda.FloatTensor')
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

    def __init__(self, numLabels=4, channel='channel_first'):
        super().__init__( )
        # self.smooth = torch.tensor(1., dtype=torch.float32)
        self.one = torch.tensor(1., dtype=torch.float32).cuda()
        self.activation = torch.nn.Softmax2d()
        self.numLables = numLabels
        self.channel= channel
        assert channel is 'channel_first' or channel == 'channel_last', r"channel has to be 'channel_first' or ''channel_last"

    @staticmethod
    def dice_loss(predict, target, smooth=1.):
        intersection = predict.contiguous().view(-1) *  target.contiguous().view(-1)
        score = (intersection.sum() * 2. + smooth) / (predict.sum() + target.sum()  + smooth)
        return 1. - score

    def forward(self, predict, target):
        dice = 0
        predict = self.activation(predict)
        if self.channel == 'channel_first':
            for c in range(self.numLabels):
                # Lme = [0.1, 0.1, 0.3, 0.5]
                temp = self.dice_loss(predict[:, c, :, :], target[:, c, :, :])
                dice += temp
        else:
            for c in range(self.numLabels):
                # Lme = [0.1, 0.1, 0.3, 0.5]
                temp = self.dice_loss(predict[:, :, :, c], target[:, :, :, c])
                dice += temp

        dice = dice /self.numLabels

        # entropy = entropy_loss(y_true) + ent
        return dice



class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, classes=4, sigmoid_normalization=True, skip_index_after=None, epsilon=1e-6,):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.classes = None
        self.skip_index_after = None
        self.register_buffer('weight', weight)
        self.epsilon = epsilon
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

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
