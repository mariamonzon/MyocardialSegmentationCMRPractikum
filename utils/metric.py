import numpy as np
# from medpy.metric.binary import hd, dc, asd
from utils.loss import DiceCoefMultilabelLoss

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def dice_coefficient_numpy(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1.0) / (np.sum(y_true) + np.sum(y_pred) + 1.0)


def dice(pred, targs):
    pred = pred.flatten()
    targs = targs.flatten()
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


def dice_coefficient_multiclass( y_pred, y_true):
    dice_metric = 0
    for c in range(y_true.shape[1]):
        dice_metric += DiceCoefMultilabelLoss.dice_coeff( predict = y_pred[:, c, :, :], target= y_true[:, c, :, :])
    dice_metric /= (c)
    return dice_metric


def hausdorff_multilabel(y_true, y_pred, numLabels=4, channel='channel_first'):
    """
    :param y_true:
    :param y_pred:
    :param numLabels:
    :return:
    """
    assert channel=='channel_first' or channel=='channel_last', r"channel has to be either 'channel_first' or 'channel_last'"
    hd_score = 0
    if channel == 'channel_first':
        y_true = np.moveaxis(y_true, 1, -1)
        y_pred = np.moveaxis(y_pred, 1, -1)
    for index in range(1, numLabels):
        temp = hd(reference=y_true[:, :, :, index], result=y_pred[:, :, :, index])
        hd_score += temp

    hd_score = hd_score / (numLabels - 1)
    return hd_score

def metrics(img_gt, img_pred, apply_hd=False, apply_asd=False):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = {}
    class_name = ["lv", "myo", "rv"]
    # Loop on each classes of the input images
    for c, cls_name in zip([1, 2, 3], class_name) :
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        h_d, a_sd = 0, 0
        if apply_hd:
            h_d = hd(gt_c_i, pred_c_i)
        if apply_asd:
            a_sd = asd (gt_c_i, pred_c_i)

        # Compute volume
        res[cls_name] = [dice, h_d, a_sd]

    return res
