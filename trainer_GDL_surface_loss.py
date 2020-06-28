"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from utils.utils import one_hot_mask
from utils.loss import DiceCoefMultilabelLoss, LossMeter, GDiceSurfaceLoss
from model.dilated_unet import Segmentation_model
from model.lr_finder import LRFinder
from utils.callbacks import EarlyStoppingCallback, ModelCheckPointCallback
from utils.metric import dice, dice_coefficient_multiclass
from dataset import MyOpsDataset
from torch.utils.data import  DataLoader
from pathlib import Path

class TrainerDistanceLoss:
    def __init__(self, model,
                 train_path="./input/images_masks_full.csv",
                 data_dir = "./input/",
                 validation_path= "",
                 IDs="",
                 valid_id="",
                 width=256,
                 height=256,  #image size
                 batch_size=4,
                 n_epoch=200,
                 gpu = True,
                 loss =  GDiceSurfaceLoss(numLabels=5),
                 n_classes = 5,
                 lr= 0.001,
                 apply_scheduler=True,  # learning rates
                 transform=False,
                 augmentation=False,
                 model_name='unet_model_checkpoint.pth.tar',
                 model_dir = '/weights/',
                 modality = ['CO', 'DE', 'T2']):

        self.train_path = Path(__file__).parent.joinpath(train_path)
        assert  self.train_path .is_file(), r"The training file_paths is not found"
        self.val_path = Path(__file__).parent.joinpath(validation_path) if Path(__file__).parent.joinpath(validation_path).is_file() else self.train_path
        self.WIDTH, self.HEIGHT = width, height
        self.BATCH_SIZE = batch_size
        self.augmentation = augmentation
        self.epochs = n_epoch
        # Set the network parameters
        self.device = 'cuda' if gpu else 'cpu'
        self.net = model.to(self.device)
        self.lr = lr
        self.optim  = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.99))
        lr_scheduler = ReduceLROnPlateau(optimizer=self.optim, mode='max', factor=.1, patience=100, verbose=True)
        self.lr_scheduler =  lr_scheduler if  apply_scheduler else False
        self.loss = loss.to(self.device)
        self.n_classes = n_classes
        self.to_save_entire_model = False # with model structure
        self.model_name = model_name
        self.model_dir = model_dir
        self.logs = 50
        # record train metrics in tensorboard and save in folder /run_logs
        self.writer = SummaryWriter( log_dir= Path(__file__).parent.absolute()/'run_logs', comment=self.model_name)
        self.loss_logs = { 'train_loss': [], 'train_dice' : [], 'val_loss': [], 'val_dice' : []}

        # Set the datasets
        self.train_dataset = MyOpsDataset(self.train_path, data_dir, augmentation=  transform,
                                          series_id=IDs.astype(str),
                                          split= True,
                                          phase = 'train',
                                          image_size = (self.WIDTH, self.HEIGHT),
                                          n_classes=n_classes,
                                          modality = modality)
        train_params = {'batch_size': batch_size, 'shuffle': True} #, 'num_workers': 4}
        self.train_dataloader = DataLoader(self.train_dataset, ** train_params)
        self.val_dataloader = DataLoader(MyOpsDataset(self.val_path, data_dir,
                                                      split= True,
                                                      series_id= valid_id.astype(str),
                                                      phase = 'valid',
                                                      image_size =  (self.WIDTH, self.HEIGHT),
                                                      n_classes=n_classes,
                                                      modality = modality),
                                        batch_size=1, shuffle= False)

        # Early stop with regards to the validation multi dice Coefficient LOSS
        self.earlystop = EarlyStoppingCallback(patience=15, mode="min")

        # Early stop with regards to the validation Dice coeffficient
        self.checkpoint = ModelCheckPointCallback(mode="max",
                                                  model_path=  Path(__file__).parent.absolute() / self.model_dir,
                                                  model_name = self.model_name,
                                                  entire_model=self.to_save_entire_model)


    def find_learning_rate(self):
        r""" Function that enables a pretraining for determining the LR
        """
        lr_finder = LRFinder(self.net, self.train_dataset, criterion=self.loss, device=self.device)
        lr_finder.range_test(self.train_dataloader, end_lr=0.1, num_iter=150, step_mode="exp")
        lr_finder.plot()

    def train_epoch(self, epoch):
        # train unet
        loss_meter = LossMeter()
        dice_metric =  LossMeter()

        self.net.train()
        # pid = os.getpid()
        for iter, data in enumerate(self.train_dataloader):
            image , mask= data['image'].to(self.device), data['mask'].to(self.device)
            distance = data['distance_map'].to(self.device)
            self.optim.zero_grad()
            output = self.net(image)
            output_probs = nn.Softmax2d()(output)
            loss = self.loss(output_probs, mask, distance)   # loss =  self.loss(output , mask)
            loss.backward()
            self.optim.step()
            # Update loss recorder tracker metrics
            loss_meter.update(loss.item(), output.size(0) )
            l = dice_coefficient_multiclass(output_probs, mask).item()
            dice_metric.update(l, output.size(0))
            if iter % self.logs == 0: # Print logs
                print('Epoch [{0}][{1}/{2}]:\t' 'Loss {loss:.4f} '.format(epoch, iter, len(self.train_dataloader), loss=loss.item() ))
            del image, mask, distance, output,output_probs, l
        train_loss = loss_meter.get_avg_loss()
        self.loss_logs['train_loss'].append( loss_meter.get_avg_loss())
        self.loss_logs['train_dice'].append(dice_metric.get_avg_loss())
        print('Epoch: [{0}]\t\t' 'Mean Train Loss:   {1:.5f} \t   Dice:  {2:.5f} \n'.format(epoch, train_loss,  dice_metric.get_avg_loss()))
        torch.cuda.empty_cache()
        return train_loss



    def validation(self):
        self.net.eval()
        loss_meter = LossMeter()
        dice_metric =  LossMeter()
        with torch.no_grad():
            for data in self.val_dataloader:
                image, mask= data['image'].to(self.device), data['mask'].to(self.device)
                distance =  data[ 'distance_map'].to(self.device)
                self.optim.zero_grad()
                output = self.net(image)
                output_probs = nn.Softmax2d()(output)
                loss = self.loss(output_probs, mask, distance)
                loss_meter.update(loss.item())
                output_mask = one_hot_mask(output_probs, channel_axis=1)
                l = dice_coefficient_multiclass(output_mask, mask).item()
                dice_metric.update(l, output.size(0))
            del image, mask, distance, output, output_probs, l
            torch.cuda.empty_cache()
        val_loss = loss_meter.get_avg_loss()
        self.loss_logs['val_loss'].append( loss_meter.get_avg_loss())
        self.loss_logs['val_dice'].append( dice_metric.get_avg_loss())
        print('Validation: \t Mean Loss:  {0:.5f}   \t    Dice:  {1:.5f}   \n'.format( val_loss , dice_metric.get_avg_loss()))
        return loss_meter.get_avg_loss(),  dice_metric.get_avg_loss()

    def train_model(self):

        for epoch in range(self.epochs):
            print(20*'+'+' Epoch {} '.format(epoch)+ 20*'+')
            #########   TRAINING   ################
            epoch_loss = self.train_epoch(epoch)
            #########   VALIDATION   ################
            val_loss_epoch, dice_val = self.validation()

            # reduceLROnPlateau (Should be applied on validation loss)
            if self.lr_scheduler:
                self.lr_scheduler.step(metrics= self.loss_logs['val_loss'][-1])

            # model check point
            self.checkpoint.step(monitor=dice_val, model=self.net, epoch=epoch)

            # early stop
            self.earlystop.step(self.loss_logs['val_loss'][-1])
            if self.earlystop.should_stop():
                break

        best_epoch = self.checkpoint.epoch

        print("Best model on epoch {}: train_dice {}, valid_dice {}".format(best_epoch,
                                                                            self.loss_logs['train_loss'][best_epoch],
                                                                            self.loss_logs['val_dice'][best_epoch]))

        i = 0
        print("Write a Tensorboard Training Summary")
        for t_loss,  v_loss, dice in  zip(self.loss_logs['train_loss'],self.loss_logs['val_loss'], self.loss_logs['val_dice']):
            self.writer.add_scalar('Loss/Training', t_loss, i)
            self.writer.add_scalar('Loss/Validation', v_loss, i)
            self.writer.add_scalar('Dice/Validation', v_loss, i)
            i += 1
        self.writer.close()
        # Return the best validation Diece metric
        return  self.loss_logs['val_dice'][best_epoch]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", help="set the learning rate for the unet", type=float, default=0.0001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=100)
    parser.add_argument("-da", "--augmentation", help="whether to apply data augmentation",default=False)
    parser.add_argument("-gpu",  help="Set the device to use the GPU", type=bool, default= False)
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=-1)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=4)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=5)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for Unet", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=4)
    parser.add_argument("-pt", "--pretrained", help="whether to train from scratch or resume", action="store_true",
                        default=False)
    parser.add_argument("-lr_find",  help="Run a pretraining to save the optimal lr", type=bool, default=False)
    parser.add_argument("-mod",  help="MRI modality: 0-all, 1-C0, 2-DE, 3-T2, 4-channelwise", type=int, default=4)

    args = parser.parse_args()

    config_info = "filters {}, n_block {}".format(args.n_filter, args.n_block)
    print(config_info)

    MR = [['multi'], ['CO'], ['DE'], ['T2'], ['CO', 'DE', 'T2']]
    alpha = 0.01
    # torch.cuda.current_device()
    CV = 5
    CV_dice = CV*[None]
    IDS = np.arange(101,126)
    MODALITY =  MR[args.mod]  # ['CO'] # ['CO', 'DE','T2']  MODALITY = ['CO']['DE']['T2']
    for i in range(CV):
        valid_id = IDS[5*i:5*(i+1)]
        train_id = IDS[~np.in1d( IDS, valid_id)]

        # calculate the comments
        comments = "segmentation_unet_lr_{}_{}".format( args.lr, args.n_filter)
        if args.augmentation:
            comments += "_augmentation"
        comments += "_crop_image_GD_Surface_loss_{}".format(alpha)
        comments += "_classes_{}".format( args.n_class)
        comments += '_' + '-'.join(MODALITY)
        comments += "_fold_{}".format(i)
        print(comments)

        model = Segmentation_model(filters=args.n_filter,
                                        in_channels=3,
                                        n_block=args.n_block,
                                        bottleneck_depth=4,
                                        n_class=args.n_class
                                   )
        if args.pretrained:
            model.load_state_dict(torch.load('./weights/{}/unet_model_checkpoint.pt'.format(comments)))

        train_obj = TrainerDistanceLoss(model,
                                        train_path="./input/filenames.csv",
                                        data_dir = r"./input/resampled_input_crop_128/",
                                        IDs=train_id,
                                        valid_id=valid_id,
                                        width=  128,
                                        height= 128,
                                        batch_size= args.batch_size,  # 8
                                        loss= GDiceSurfaceLoss(n_classes=args.n_class, alpha =alpha),
                                        n_classes =args.n_class,
                                        augmentation=args.augmentation,
                                        lr=args.lr,
                                        n_epoch=args.epochs,
                                        model_name= 'unet_model_checkpoint.pth.tar',
                                        model_dir = './weights/{}/'.format(comments),
                                        modality=MODALITY
                                        )

        print("The validation IDs are ", valid_id)
        # Train the models
        print("********** Training fold ", i, " ***************")
        start = datetime.now()
        torch.autograd.set_detect_anomaly(True)
        CV_dice[i] = train_obj.train_model()
        print("Time elapsed for training (hh:mm:ss.ms) {}".format( datetime.now() - start))
        del model, train_obj
        torch.cuda.empty_cache()
    print( " The best accuracies per fold are: ", CV_dice)
    exit(0)