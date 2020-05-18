"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
import torch
import torch.nn as nn
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from utils.utils import soft_to_hard_pred, hard_predicton
from utils.loss import DiceCoefMultilabelLoss, LossMeter
from model.dilated_unet import Segmentation_model
from model.lr_finder import LRFinder
from utils.callbacks import EarlyStoppingCallback, ModelCheckPointCallback
from utils.metric import dice_coefficient_multiclass
from dataset import MyOpsDataset
from torch.utils.data import  DataLoader
from pathlib import Path


class Trainer:
    def __init__(self, model,
                 train_path="./input/images_masks_full.csv",
                 data_dir = "./input/",
                 validation_path= "",
                 IDs="",
                 valid_id="",
                 width=256,
                 height=256,  #image size
                 batch_size=4,
                 n_epoch=100,
                 gpu = False,
                 loss = DiceCoefMultilabelLoss(numLabels=4),
                 lr= 0.001,
                 apply_scheduler=True,  # learning rates
                 transform=False,
                 gaussian_noise=False,
                 model_name='unet_model_checkpoint.pt',
                 model_dir = './weights/'  ,
                 summary_name='./summary/',
                 channel='channel_first'):

        assert channel is 'channel_first' or channel == 'channel_last', r"channel has to be 'channel_first' or ''channel_last"
        assert Path(train_path).is_file(), r"The training file_paths is not found"
        self.train_path = Path(train_path)
        self.val_path =  validation_path if Path(validation_path).is_file() else self.train_path

        self.WIDTH, self.HEIGHT = width, height
        self.BATCH_SIZE = batch_size
        self.noise = gaussian_noise
        self.epochs = n_epoch

        # Set the network parameters
        self.device = 'cuda' if gpu else 'cpu'
        self.net = model.to(self.device)
        self.lr = lr
        self.optim  = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.99))
        lr_scheduler = ReduceLROnPlateau(optimizer=self.optim, mode='max', factor=.1, patience=100, verbose=True)
        self.lr_scheduler =  lr_scheduler if  apply_scheduler else False
        self.model_name = model_name
        self.channel = channel
        self.loss = loss.to(self.device)
        self.to_save_entire_model = False # with model structure
        self.file_to_save_summary = summary_name
        self.logs = 0
        # Set the datasets
        self.train_dataset = MyOpsDataset(train_path, data_dir, transform =  transform,
                                          series_id=IDs.astype(str),
                                          train_val_split= True,
                                          phase = 'train',
                                          image_size = (self.WIDTH, self.HEIGHT),
                                          modality = 'T2')
        train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4}
        self.train_dataloader = DataLoader(self.train_dataset, ** train_params)
        self.val_dataloader = DataLoader(MyOpsDataset(self.val_path, data_dir,
                                                      train_val_split= True,
                                                      series_id= valid_id.astype(str),
                                                      phase = 'valid',
                                                      image_size =  (self.WIDTH, self.HEIGHT),
                                                      modality = 'T2'),
                                         batch_size=1, shuffle= False)

        self.earlystop = EarlyStoppingCallback(patience=10, mode="max")
        self.checkpoint = ModelCheckPointCallback(mode="max",
                                                  model_path=model_dir,
                                                  model_name = self.model_name,
                                                  entire_model=self.to_save_entire_model)

    def find_learning_rate(self):
        r""" Function that enables a pretraining for determining the LR
        """
        lr_finder = LRFinder(self.net, self.train_dataset, criterion=self.loss, device=self.device)
        lr_finder.range_test(self.train_dataloader, end_lr=0.1, num_iter=150, step_mode="exp")
        lr_finder.plot()


    def validation(self):

        self.net.eval()
        loss_meter = LossMeter()
        dice_loss =  LossMeter()
        with torch.no_grad():
            for data in self.train_dataloader:
                image = data['image'].to(self.device)  # Input images
                mask =  data['mask'].to(self.device)
                output = self.net(image, features_out=False)
                loss = self.loss(output, mask)
                loss_meter.update(loss.item())

                y_pred = hard_predicton(output, channel=1)
                l = dice_coefficient_multiclass(mask, y_pred, numLabels=4).item()
                dice_loss.update(l.item(), output.size(0))

        train_loss = loss_meter.get_avg_loss()
        self.loss_logs['cal_loss'].append( loss_meter.get_avg_loss())
        self.loss_logs['val_dice'].append( dice_loss.get_avg_loss())
        print('Average val_loss: {0:.5f} '.format(train_loss))
        print('Average val_dice: {  dice_loss:.5f} '.format(dice_loss=self.loss_logs['val_dice'][-1]))

        return loss_meter.get_avg_loss(),  dice_loss.get_avg_loss(),


    def toggle_phase(self, phase="train"):
        assert phase == "train" or phase == "eval"
        if phase == "train":
            self.net.train()
        else:
            self.net.eval()

    #
    # def togglegrads(self, require_grads=True):
    #     for param in self.net.parameters():
    #         param.requires_grad = require_grads


    def train_epoch(self, epoch):

        loss_meter = LossMeter()
        dice_loss =  LossMeter()

        self.net.train()
        # pid = os.getpid()
        # train unet
        for iter, data in enumerate(self.train_dataloader):
            image , mask = data['image'].to(self.device), data['mask'].to(self.device)
            self.optim.zero_grad()
            segmentation, btnck = self.net(image, features_out=True)
            loss =  self.loss(segmentation , mask)
            loss.backward()
            self.optim.step()
            loss_meter.update(loss.item(), segmentation.size(0) )

            # Compute Hard Dice Loss
            y_pred = hard_predicton(segmentation, channel =1)
            l =  dice_coefficient_multiclass(mask, y_pred, numLabels=4).item()
            dice_loss.update(l.item(), segmentation.size(0))

            if iter % self.logs == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f} '.format(epoch, iter, len(self.train_dataloader), loss=loss.item() ))



        train_loss = loss_meter.get_avg_loss()
        self.loss_logs['train_loss'].append( loss_meter.get_avg_loss())
        self.loss_logs['train_dice'].append( dice_loss.get_avg_loss())
        print('Average train_loss: {0:.5f} '.format(train_loss))
        print('Average train_dice: {  dice_loss:.5f} '.format(dice_loss=self.loss_logs['train_dice'][-1]))

        return train_loss


    def train_model(self, train=True, model_name=''):

        # record train metrics in tensorboard
        self.writer = SummaryWriter(comment=self.model_name)
        self.loss_logs = { 'train_loss': [], 'train_dice' : [], 'val_loss': [], 'val_dice' : []}

        for epoch in range(self.epochs):

            #########   TRAINING   ################
            epoch_loss = self.train_epoch(epoch)
            #########   VALIDATION   ################
            val_loss_epoch = self.validation()

            # reduceLROnPlateau
            if self.lr_scheduler:
                self.lr_scheduler.step(metrics= self.loss_logs)

            # model check point
            self.checkpoint.step(monitor=val_loss_epoch, model=self.net, epoch=epoch)

            # early stop
            self.earlystop.step(self.loss_logs['val_dice'][-1])
            if self.earlystop.should_stop():
                break

        best_epoch = self.checkpoint.epoch
        print("Best model on epoch {}: train_dice {}, valid_dice {}".format(best_epoch,
                                                                            self.loss_logs['train_dice'][best_epoch],
                                                                            self.loss_logs['val_dice'][best_epoch]))


        i = 0
        print("write a training summary")
        for t_loss, t_dice, v_loss, v_dice in  zip(self.loss_logs['train_loss'],self.loss_logs['val_loss'] ):
            self.writer.add_scalar('Loss/Training', t_loss, i)
            self.writer.add_scalar('Loss/Validation', v_loss, i)
            i += 1
        self.writer.close()
        print("Finish training")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", help="set the learning rate for the unet", type=float, default=0.0001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=300)
    parser.add_argument("-gn", "--gaussianNoise", help="whether to apply gaussian noise", action="store_true",
                        default=True)
    parser.add_argument("-gpu",  help="Det the device to use the GPU", type=bool, default=False)
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=2)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=6)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for Unet", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=4)
    parser.add_argument("-pt", "--pretrained", help="whether to train from scratch or resume", action="store_true",
                        default=False)
    args = parser.parse_args()

    config_info = "filters {}, n_block {}".format(args.n_filter, args.n_block)
    print(config_info)

    # calculate the comments
    comments = "Segmentation_unet_lr_{}_{}".format(args.lr, args.n_filter)
    if args.gaussianNoise:
        comments += ".gaussian_noise"
    print(comments)

    CV = 5
    IDS = np.arange(101,126)

    for i in range(CV):
        valid_id = IDS[5*i:5*(i+1)]
        train_id = IDS[~np.in1d( IDS, valid_id)]
        print("The Train IDs are ", train_id)

        model = Segmentation_model(filters=args.n_filter,
                                        in_channels=3,
                                        n_block=args.n_block,
                                        bottleneck_depth=4,
                                        n_class=args.n_class)

        if args.pretrained:
            model.load_state_dict(torch.load('./weights/{}/unet_model_checkpoint.pt'.format(comments)))

        train_obj = Trainer( model,
                            train_path="./input/images_masks_full.csv",
                            data_dir = "./input/",
                             IDs=train_id,
                            valid_id=valid_id,
                            width= 32,
                            height=32,
                            batch_size=args.batch_size,  # 8
                            loss=DiceCoefMultilabelLoss(numLabels=args.n_class),
                            gaussian_noise=args.gaussianNoise,
                            lr=args.lr,
                            n_epoch=args.epochs,
                            )

        # Train the models
        print(" Training fold ", i)
        start = datetime.now()
        torch.autograd.set_detect_anomaly(True)
        train_obj.train_model(model_name=comments)
        end = datetime.now()
        print("time elapsed for training (hh:mm:ss.ms) {}".format(end - start))