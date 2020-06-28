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
from utils.metric import dice_coefficient_multiclass, dice
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from dataset import MyOpsDataset
import pandas as pd
from skimage import measure


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
            # dataset.save_crop_images(categorical_mask2image(output_masks)[0] , d)
            # output_masks[0]  = torch.from_numpy( keep_largest_connected_components(output_masks.cpu().numpy()[0] ))
            dice_accuracy = dice_coefficient_multiclass(output_masks, mask).item()
            scar_dice = dice(output_masks[:,-2], mask[:,-2]).item()
            edema_dice = dice(output_masks[:, -1], mask[:, -1]).item()
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


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in  [1,2,3,4,5]:

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

    name_folder = 'segmentation_unet_lr_0.0001_32_crop_image_surface_loss_0.01_classes_5_CO-DE-T2'

    args.n_class = int(name_folder.split('_')[-2])
    modality = name_folder.split('_')[-1].split('-')
    model = Segmentation_model(filters=args.n_filter,
                                    in_channels=3,
                                    n_block=args.n_block,
                                    bottleneck_depth=4,
                                    n_class=args.n_class)
    # modality = ['multi']
    for FOLD in range(0,5):
        model_name = name_folder + '_fold_{}'.format(FOLD) #args.model_name
        # FOLD = int(model_name[-1])
        #['CO'] [, 'DE', 'T2']
        print(model_name)

        model.load_state_dict(torch.load('./weights/{}/unet_model_checkpoint.pth.tar'.format(model_name)))

        test_set= "original_crop_128"
        valid_id = np.arange(101,126)[5 * FOLD:5 * (FOLD + 1)]
        dataset = MyOpsDataset("./input/images_masks_modalities.csv", "./input/"+test_set,
                                              split= True,
                                              series_id= valid_id.astype(str),
                                              phase = 'valid',
                                              image_size =  (128, 128),
                                              n_classes=args.n_class,
                                              modality = modality, crop_center=0)


        result_dir = make_directory( './results/'+model_name, test_set )
        dice_accuracy = predict_model(dataset, model, device='cpu', save_images= True, dir_name= result_dir)
        metrics = pd.DataFrame(dice_accuracy, columns=['Dice coefficient'])
        metrics.to_excel( './results/{}/test_results.xlsx'.format(model_name))