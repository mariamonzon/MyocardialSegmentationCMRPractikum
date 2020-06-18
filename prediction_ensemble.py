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
from model.dilated_unet import Ensemble_model, Segmentation_model
import argparse
from utils.utils import one_hot_mask, categorical_mask2image
from utils.utils import make_directory
from dataset import MyOpsDataset
import torch.nn as nn
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


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1,2,3]: # [1,2,3,4,5]:

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


def compute_metrics_on_files(gt, pred, dir_name=''):
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

    f = open('{}/output.txt'.format(dir_name), 'a')
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


def load_image(self,idx ):
    key = self.file_names.columns[self.modality[0]]
    image = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx][key], mode='RGB'))
    if len(self.modality)>1:
        # key = self.file_names.columns[0]
        for i, m in enumerate(self.modality):
            key = self.file_names.columns[m]        # key ='img_' + m
            image[:,:, i] = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx][ key ], mode='L'))
    return image

def read_img(pat_id, img_len, type='C0'):
    images=[]
    for im in range(img_len):
        # img = MyOpsDataset.PIL_loader(r'./input/processed/train/myops_training_{}_{}_{}.png'.format(pat_id, type, im))
        if type == 'multi' or  type == 'CO-DE-T2':
            img = cv2.imread(r'./input/train/myops_training_{}_C0_{}.png'.format(pat_id, type, im))
            img[1] = cv2.imread(r'./input/train/myops_training_{}_DE_{}.png'.format(pat_id, type, im), cv2.IMREAD_GRAYSCALE)
            img[2] = cv2.imread(r'./input/train/myops_training_{}_T2_{}.png'.format(pat_id, type, im), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(r'./input/train/myops_training_{}_{}_{}.png'.format(pat_id, type, im))
        images.append(img)
    return np.array(images)


def evaluate_segmentation(fold=0,   device = 'cpu', model_name = 'unet', mod = 'C0' ):
    """
    :param Model_name: Name of the trained model
    """
    dir_path = make_directory('results',  model_name )
    print(dir_path)
    unet_model.eval()
    ids =  np.arange(101+5* fold, 101+ 5 * (fold + 1))
    metrics = []
    with torch.no_grad():
        for pat_id in ids:
            test_path = sorted(glob("input/raw/train/myops_training_{}_{}.nii.gz".format(pat_id, mod)))
            mask_path = sorted(glob("input/raw/masks/myops_training_{}_gd.nii.gz".format(pat_id)))
            for imgPath, mskPath in zip(test_path, mask_path):
                nimg, affine, header = load_nii(mskPath)
                # print(nimg.shape)
                vol_resize = read_img(pat_id, nimg.shape[2], mod)
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

                metrics.append(compute_metrics_on_files(nimg, pred, dir_path))
                del pred, nimg
    metrics = np.asarray(metrics).mean(axis=0)
    return metrics

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
    results = np.zeros((5,15))
    for fold in range(5):
        unet_model = Ensemble_model(filters=args.n_filter,
                               in_channels=3,
                               n_block=args.n_block,
                               bottleneck_depth=4,
                               n_class=args.n_class
                               )

        device = 'gpu' if args.gpu else 'cpu'
        unet_model.to(device)
        mod = 'CO-DE-T2'
        model_name ='ensemble_unet_lr_0.0001_32_{}_fold_{}'.format( mod, fold)
        unet_model.load_state_dict(torch.load('weights/'+model_name+'/unet_model_checkpoint.pth.tar'), strict=False)
        print("model loaded:  ", model_name)
        results[fold] = evaluate_segmentation(fold, device, model_name)

        del unet_model
    results = ["{:.3f}".format(r) for r in results.mean(axis=0)]
    print(results)