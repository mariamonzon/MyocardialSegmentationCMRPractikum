"""
@Author: Sulaiman Vesal, Maria Monzon
"""

import numpy as np
from glob import  glob
import cv2
import torch
from medpy.metric.binary import hd, dc, asd
import nibabel as nib
from skimage import measure
from model.dilated_unet import Ensemble_model
import argparse
from utils.utils import one_hot_mask, categorical_mask2image
from dataset import MyOpsDataset
import torch.nn as nn

import SimpleITK as sitk
import os
from multiprocessing import pool
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2


def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')


def preprocess_image(image, spacing, is_seg=False, spacing_target=(1, 0.5, 0.5), keep_z_spacing=False):


    if not is_seg:
        order_img = 1
        image = resize_image(image, spacing, spacing_target, order=order_img).astype(np.float32)
        return image
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        print(vals)
        results = []
        for i in range(len(tmp)):
            results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
        image = vals[np.vstack(results).argmax(0)]
    return image

#
# Functions to process files, directories and metrics
#

def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1,2,3,4,5]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img


def metrics(img_gt, img_pred):
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

    res = []
    # Loop on each classes of the input images
    for c in [200, 500, 600, 1220, 2221]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)
        pred_c_i[0,0,0] = 1 if len(np.unique( pred_c_i)) ==1 else pred_c_i[0,0, 0]

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        h_d = hd(gt_c_i, pred_c_i)
        a_sd = asd (gt_c_i, pred_c_i)

        # Compute volume
        res += [dice, h_d, a_sd]

    return res


def compute_metrics_on_files(gt, pred):
    """
    Function to give the metrics for two files

    Parameters
    ----------

    path_gt: string
    Path of the ground truth image.

    path_pred: string
    Path of the predicted image.
    """
    res = metrics(gt, pred) ; res_rtu = res
    res = ["{:.3f}".format(r) for r in res]

    formatting = "{:>8} , {:>8} , {:>8} , {:>8} , {:>8} , {:>8} , {:>8} , {:>8} , {:>8}"
    # print(formatting.format(*HEADER))
    # print(formatting.format(*res))

    f = open('output.txt', 'a')
    print(formatting.format(*res), file=f)  # Python 3.x

    return res_rtu


def crop_volume(vol, crop_size=112):
    """
    :param vol:
    :return:
    """

    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size,: ])


def reconstuct_volume(vol, img_shape, crop_size=112):
    """
    :param vol:
    :return:
    """
    recon_vol = np.zeros((len(vol), img_shape[0], img_shape[1],img_shape[2]), dtype=np.float32)

    recon_vol[:,
    int(recon_vol.shape[1] / 2) - crop_size: int(recon_vol.shape[1] / 2) + crop_size,
    int(recon_vol.shape[2] / 2) - crop_size: int(recon_vol.shape[2] / 2) + crop_size, :] = vol

    return recon_vol


def read_img(pat_id, img_len, type='C0'):
    images = []
    for im in range(img_len):
        # img = MyOpsDataset.PIL_loader(r'./input/processed/train/myops_training_{}_{}_{}.png'.format(pat_id, type, im))
        img = cv2.imread(r'./input/train/myops_training_{}_{}_{}.png'.format(pat_id, type, im))
        images.append(img)
    return np.array(images)


def evaluate_segmentation( device = 'cpu'):
    """
    :param Model_name: Name of the trained model
    """

    with torch.no_grad():
        for pat_id in range(106, 126):
            test_path = sorted(glob("input/raw/train/myops_training_{}_{}.nii.gz".format(pat_id, 'C0')))
            mask_path = sorted(glob("input/raw/masks/myops_training_{}_gd.nii.gz".format(pat_id)))
            for imgPath, mskPath in zip(test_path, mask_path):
                itk_image = sitk.ReadImage(test_path)
                spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
                nimg, affine, header = load_nii(mskPath)
                # print(nimg.shape)
                vol_resize = read_img(pat_id, nimg.shape[2])
                x_batch = np.array(vol_resize, np.float32) / 255.
                x_batch = np.moveaxis(x_batch, -1, 1)
                # print(x_batch.shape)
                pred= unet_model(torch.tensor(x_batch).to(device))
                pred = one_hot_mask(pred)
                pred = categorical_mask2image(pred)
                pred = np.moveaxis(pred, 1, 3)
                pred_resize = reconstuct_volume(pred, nimg.shape, crop_size=128)
                # print("Prediction shape", pred.shape)
                pred = np.stack(np.array(pred_resize), axis=3)
                pred = np.argmax(pred, axis=3)

                pred = keep_largest_connected_components(pred)
                pred = np.array(pred).astype(np.uint16)
                pred = np.where(pred == 1, 200, pred)
                pred = np.where(pred == 2, 500, pred)
                pred = np.where(pred == 3, 600, pred)
                pred = np.where(pred == 4, 1220, pred)
                pred = np.where(pred == 5, 2221, pred)
                pred = preprocess_image(pred, spacing= (2.0, 1.0, 1.0), is_seg=False, spacing_target=spacing, keep_z_spacing=False)
                compute_metrics_on_files(nimg, pred)

                save_nii(img_path=mskPath, data=pred, affine=affine, header=header )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", help="set the learning rate for the unet", type=float, default=0.0001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=300)
    parser.add_argument("-da", "--augmentation", help="whether to apply data augmentation",default=False)
    parser.add_argument("-gpu",  help="Set the device to use the GPU", type=bool, default=False)
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=4)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=6)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for Unet", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=4)
    parser.add_argument("-pt", "--pretrained", help="whether to train from scratch or resume", action="store_true",
                        default=False)
    parser.add_argument("-lr_find",  help="Run a pretraining to save the optimal lr", type=bool, default=False)
    parser.add_argument("-mod",  help="MRI modality: 0-all, 1-C0, 2-DE, 3-T2, 4-channelwise", type=int, default=0)

    args = parser.parse_args()
    config_info = "filters {}, n_block {}".format(args.n_filter, args.n_block)
    print(config_info)

    unet_model = Ensemble_model(filters=args.n_filter,
                           in_channels=3,
                           n_block=args.n_block,
                           bottleneck_depth=4,
                           n_class=args.n_class
                           )
    device = 'gpu' if args.gpu else 'cpu'
    unet_model.to(device)
    unet_model.load_state_dict(torch.load('weights/ensemble_unet_lr_0.0001_32_CO-DE-T2_fold_0/unet_model_checkpoint.pth.tar'), strict=False)
    print("model loaded")
    evaluate_segmentation(device)