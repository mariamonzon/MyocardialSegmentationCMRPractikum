# The code is partially adapted from the following github link under the copyright:
# https://github.com/MIC-DKFZ/ACDC2017/blob/master/dataset_utils.py
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)


import SimpleITK as sitk
import os
from multiprocessing import pool
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2


def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


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


def crop_volume(vol, crop_size=128):

    """
    :param vol:
    :return:
    """

    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


def preprocess_volume(img_volume):

    """
    :param img_volume: A patient volume
    :return: applying CLAHE and Bilateral filter for contrast enhacnmeent and denoising

    """
    prepross_imgs = []
    for i in range(len(img_volume)):
        img = img_volume[i]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        prepross_imgs.append(cl1)
    return np.array(prepross_imgs)

def process_patient(id, type='T2'):

    fname = "../input/raw/train/myops_training_{}_{}.nii.gz".format(id, type)
    itk_image = sitk.ReadImage(fname)
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    itk_image = sitk.Cast(sitk.RescaleIntensity(itk_image), sitk.sitkUInt8)
    images = sitk.GetArrayFromImage(itk_image)
    images = preprocess_volume(images)

    fname =  "../input/raw/masks/myops_training_{}_gd.nii.gz".format(id, type)
    itk_masks = sitk.ReadImage(fname)
    itk_masks = sitk.Cast(sitk.RescaleIntensity(itk_masks), sitk.sitkUInt8)
    masks = sitk.GetArrayFromImage(itk_masks)

    masks = np.where(masks == 22, 1, masks)
    masks = np.where(masks == 57, 2, masks)
    masks = np.where(masks == 68, 3, masks)
    masks = np.where(masks == 140, 4, masks)
    masks = np.where(masks == 255, 5, masks)

    #print(np.unique(masks))
    images = preprocess_image(images, spacing, spacing_target=(2.0, 1.0, 1.0),  keep_z_spacing=False)
    masks = preprocess_image(masks, spacing, is_seg=True, spacing_target=(2.0, 1.0, 1.0), keep_z_spacing=False)

    #images = crop_volume(images)
    #masks = crop_volume(masks)

    #print(id, images.shape, masks.shape, masks.dtype, images.dtype)

    l =0
    for i, n in zip(images, masks):
        plt.imsave( '../input/processed/trainAresampled/myops_training_{}_{}_{}.png'.format(id, type, l), i, cmap='gray')
        print('../input/processed/masksAresampled/myops_training_{}_{}_{}.png'.format(id, type, l),np.unique(n))
        #plt.imsave('../input/processed/masksAresampled/myops_training_{}_{}_gd_{}.png'.format(id, type, l), n, cmap='gray')
        np.save('../input/processed/masksAresampled/myops_training_{}_{}_gd_{}.npy'.format(id, type, l), n)
        l+=1

    # print(images.shape, masks.shape)
    # plt.imshow(images[10, :, :], cmap='gray')
    # plt.show()
    # plt.imshow(masks[10, :, :], cmap='gray')
    # plt.show()


if __name__ == "__main__":

    for i in range(101, 126):
      process_patient(i)
