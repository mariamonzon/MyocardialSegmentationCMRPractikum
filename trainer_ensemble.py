"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
import torch

import numpy as np
from datetime import datetime
import argparse
from utils.loss import DiceCoefMultilabelLoss, LossMeter, DiceLoss
from model.dilated_unet import Ensemble_model
from trainer import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", help="set the learning rate for the unet", type=float, default=0.0001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=300)
    parser.add_argument("-da", "--augmentation", help="whether to apply data augmentation",default=False)
    parser.add_argument("-gpu",  help="Set the device to use the GPU", type=bool, default=True)
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
    torch.cuda.current_device()
    CV = 5
    CV_dice = CV*[None]
    IDS = np.arange(101,126)
    MODALITY =  ['CO', 'DE', 'T2']
    for i in range(CV):
        valid_id = IDS[5*i:5*(i+1)]
        train_id = IDS[~np.in1d( IDS, valid_id)]

        # calculate the comments
        comments = "ensemble_unet_lr_{}_{}".format( args.lr, args.n_filter)
        if args.augmentation:
            comments += "_augmentation"
        comments += '_' + '_'.join(MODALITY)
        comments += "_fold_{}".format(i)
        print(comments)

        model = Ensemble_model(filters=args.n_filter,
                                        in_channels=3,
                                        n_block=args.n_block,
                                        bottleneck_depth=4,
                                        n_class=args.n_class
                                   )
        if args.pretrained:
            model.load_state_dict(torch.load('./weights/{}/unet_model_checkpoint.pt'.format(comments)))


        train_obj = Trainer(model,
                            train_path="./input/images_masks_modalities.csv",
                            data_dir = "./input/",
                            IDs=train_id,
                            valid_id=valid_id,
                            width=  256,
                            height= 256,
                            batch_size= args.batch_size,  # 8
                            loss= DiceLoss(n_classes=args.n_class),
                            n_classes =args.n_class,
                            augmentation=args.augmentation,
                            lr=args.lr,
                            n_epoch=args.epochs,
                            model_name= 'unet_model_checkpoint.pth.tar',
                            model_dir = './weights/{}/'.format(comments),
                            modality = MODALITY
                            )

        if args.lr_find:
            Trainer.find_learning_rate()
            print("The learning rate finder has finished ", valid_id)
            exit(0)

        print("The validation IDs are ", valid_id)
        # Train the models
        print("********** Training fold ", i, " ***************")
        start = datetime.now()
        torch.autograd.set_detect_anomaly(True)
        CV_dice[i] = train_obj.train_model()
        print("Time elapsed for training (hh:mm:ss.ms) {}".format( datetime.now() - start))
        del model, train_obj
        torch.cuda.empty_cache()
    print( " the best accuracies per fold are: ", CV_dice)
    exit(0)