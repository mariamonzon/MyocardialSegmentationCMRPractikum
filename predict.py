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
            dataset.save_crop_images(categorical_mask2image(output_masks)[0] , d)
            dice_accuracy = dice_coefficient_multiclass(output_masks, mask).item()
            print("Image [{0}]:  \t Dice score {1:3f}".format(d, dice_accuracy))
            dice_metric[d] =  dice_accuracy
            if save_images:
                save_output_image(image, output_masks, mask, d, dice=dice_accuracy, dir_name =dir_name)
    print("Test mean acuraccy: \t {0:3f}]:".format(np.mean(dice_metric)) )
    return dice_metric

def save_output_image(image, output, mask, id, dice = None, dir_name ='./results/'):
    # plt.rc('text', usetex=True)
    image = image[0,0].cpu().numpy()
    output_image = categorical_mask2image(output)[0,0]
    mask_image = categorical_mask2image(mask)[0,0]
    f = plt.figure(figsize=(19.2,6.4), dpi =150)
    f.add_subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('CARD axial MRI Slice')
    plt.axis('off')

    f.add_subplot(1, 3, 2)
    plt.imshow( output_image , cmap='jet'),
    plt.title('Prediction Mask')
    if dice is not None:
        plt.xlabel('Dice: {0:.5f}'.format(dice))
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    f.add_subplot(1, 3, 3)
    plt.imshow( mask_image, cmap='jet'),
    plt.title('Ground Truth Mask')
    plt.axis('off')
    # plt.show(block=True)
    rect = [0, 0.03, 1, 0.95]
    f.savefig( str(dir_name) + '/image_{}.png'.format(str(id).zfill(3)))
    plt.imsave( str(dir_name) + '/mask_{}.png'.format(str(id).zfill(3)), output_image)
    # plt.imsave( str(dir_name) +  '/gt_{}.png'.format(str(id).zfill(3)), mask_image)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", help="Set the device to use the GPU", type=bool, default=False)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=2)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for Unet", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=4)
    parser.add_argument("--model_name", help="path to the model", type=str, default='segmentation_unet_lr_0.0001_32_multi_fold_0')

    args = parser.parse_args()

    config_info = "filters {}, n_block {}".format(args.n_filter, args.n_block)
    print(config_info)


    model = Segmentation_model(filters=args.n_filter,
                                    in_channels=3,
                                    n_block=args.n_block,
                                    bottleneck_depth=4,
                                    n_class=args.n_class)

    for FOLD in range(5):
        model_name = 'segmentation_unet_lr_0.0001_32_surface_loss_01_samples_500_CO-DE-T2_fold_{}'.format(FOLD) #args.model_name
        # FOLD = int(model_name[-1])
        modality = model_name.split('_')[-3].split('-') #['CO'] [, 'DE', 'T2']
        print(model_name)

        model.load_state_dict(torch.load('./weights/binary_segmentation_circle/{}/unet_model_checkpoint.pth.tar'.format(model_name)))


        valid_id = np.arange(101,126)[5 * FOLD:5 * (FOLD + 1)]
        dataset = MyOpsDataset("./input/images_masks_modalities.csv", "./input/",
                                              split= True,
                                              series_id= valid_id.astype(str),
                                              phase = 'valid',
                                              image_size =  (256, 256),
                                              n_classes=args.n_class,
                                              modality = modality, crop_center=0)


        result_dir = make_directory( './results/', model_name)
        dice_accuracy = predict_model(dataset, model, device='cpu', save_images= True, dir_name= result_dir)
        metrics = pd.DataFrame(dice_accuracy, columns=['Dice coefficient'])
        metrics.to_excel( './results/{}/test_results.xlsx'.format(model_name))