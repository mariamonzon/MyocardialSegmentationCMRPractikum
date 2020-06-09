"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.loss import DiceCoefMultilabelLoss, LossMeter
from utils.utils import one_hot_mask, categorical_mask2image
from utils.save_data import make_directory
from model.dilated_unet import Segmentation_model
from utils.metric import dice_coefficient_multiclass
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from dataset import MyOpsDataset
import pandas as pd


def predict_model(dataset, model, device ='cpu', save_images= True, dir_name =""):
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    i=0
    model.eval()
    loss_meter = LossMeter()
    dice_metric = [None]*len(dataloader)
    with torch.no_grad():
        for d, data in enumerate(dataloader):
            image = data['image'].to(device)  # Input images
            mask =  data['mask'].to(device)
            distance_map =  data['distance_map'].to(device)

            output = model(image, features_out=False)
            # Get output probability maps
            output_probs = nn.Softmax2d()(output)
            output_masks = one_hot_mask(output_probs ,  channel_axis=1)

            dice_accuracy = dice_coefficient_multiclass(output_masks, mask).item()
            print("Image [{0}]:  \t Dice score {1:3f}".format(d, dice_accuracy))
            dice_metric[d] =  dice_accuracy
            if save_images:
                save_output_image(image, output_masks, mask, d, dir_name)
    print("Test mean acuraccy: \t {0:3f}]:".format(np.mean(dice_metric)) )
    return dice_metric

def save_output_image(image, output, mask, id, dir_name ='./results/'):
    image = image[0,0].cpu().numpy()
    output_image = categorical_mask2image(output)
    mask_image = categorical_mask2image(mask)
    f = plt.figure(figsize=(9.6,5.4))
    f.add_subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('CARD axial MRI Slice')
    plt.axis('off')

    f.add_subplot(1, 3, 2)
    plt.imshow( output_image , cmap='jet'),
    plt.axis('off')
    plt.title('Prediction Mask')

    f.add_subplot(1, 3, 3)
    plt.imshow( mask_image, cmap='gray'),
    plt.title('Ground Truth Mask')
    plt.axis('off')
    # plt.show(block=True)

    # f.savefig( str(dir_name) + '/image_{}.png'.format(str(id).zfill(3)))
    plt.imsave( str(dir_name) + '/pred_{}.png'.format(str(id).zfill(3)), output_image)
    plt.imsave( str(dir_name) +  '/gt_{}.png'.format(str(id).zfill(3)), mask_image)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", help="set the learning rate for the unet", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=300)
    parser.add_argument("-da", "--augmentation", help="whether to apply gaussian noise", action="store_true",default=True)
    parser.add_argument("-gpu", help="Set the device to use the GPU", type=bool, default=False)
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=4)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=6)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for Unet", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=4)
    parser.add_argument("-pt", "--pretrained", help="whether to train from scratch or resume", action="store_true",
                        default=False)

    args = parser.parse_args()
    FOLD = 0
    config_info = "filters {}, n_block {}".format(args.n_filter, args.n_block)
    print(config_info)

    # calculate the comments
    model_params = "segmentation_unet_lr_{}_{}".format(args.lr, args.n_filter)
    if args.augmentation:
        model_params += "_augmentation"
    model_params += "_fold_{}".format(FOLD)


    model = Segmentation_model(filters=args.n_filter,
                                    in_channels=3,
                                    n_block=args.n_block,
                                    bottleneck_depth=4,
                                    n_class=args.n_class)
    model_params = 'segmentation_unet_lr_0.001_32_augmentation_T2_fold_0'
    modality = ['T2']  #['CO', 'DE', 'T2']

    print(model_params)
    model.load_state_dict(torch.load('./results_model/{}/unet_model_checkpoint.pth.tar'.format(model_params)))


    valid_id = np.arange(101,126)[5 * FOLD:5 * (FOLD + 1)]
    dataset = MyOpsDataset("./input/images_masks_full.csv", "./input/",
                                          split= True,
                                          series_id= valid_id.astype(str),
                                          phase = 'valid',
                                          image_size =  (256, 256),
                                          modality = modality)


    result_dir = make_directory( './results/', model_params)
    dice_accuracy = predict_model(dataset, model, device='cpu', save_images= True, dir_name= result_dir)
    metrics = pd.DataFrame(dice_accuracy, columns=['Dice coefficient'])
    metrics.to_excel( './results/{}/test_results.xlsx'.format(model_params))