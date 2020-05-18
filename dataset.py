"""
@Author: Sulaiman Vesal, Maria Monzon
"""

import numpy as np
import cv2
import pandas as pd
# from skimage.exposure import match_histograms
from utils.image_proces_vis import *
from matplotlib import pyplot as plt
from albumentations import (
    Resize,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    RandomCrop,
    RandomSizedCrop,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise,
    Rotate,
    GaussianBlur,
    MotionBlur,
    Blur,
    ShiftScaleRotate,
    Normalize,
    OneOf,
    Compose,
)
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
from utils.save_data import html, make_directory
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from PIL import Image
from pathlib import Path

class MyOpsDataset(Dataset):
    def __init__(self, csv_path, root_path, transform = False, series_id = "",   train_val_split = True, phase ='train', image_size = (256, 256), modality = ['CO', 'DE', 'T2']):
        """
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: dict of transforms containing parameters
                        transform = { }
                            rotation = params.get('rotation', True)
                            noise = params.get('noise', False)
                            clahe = params.get('clahe', True)
                            contrast = params.get(' contrast ', False)
                            blur = params.get('blur', False)
                            distorsion = params.get('distorsion', False)
        """
        self.file_names = pd.read_csv(csv_path, delimiter=';')
        # if Path( series_id ).is_file():
        #     self.series_id = pd.read_csv( series_id , delimiter=';')
        if train_val_split and series_id is not "":
            self.series_id = list(series_id)
            self.file_names =self.split_idx()
        self.modality = modality if isinstance(modality, list) else [modality]
        if Path(root_path).is_absolute():
            self.root_dir =  Path(root_path)
        else:
            self.root_dir = Path.cwd().joinpath(root_path)
        self.mean, self.std = self.normalization()
        self.image_size = image_size
        self.phase = phase
        self.data_augmentation = self.set_augmentation( 0.5, image_size, data_augmentation = transform,  rotation= True, crop = True, clahe=True, noise=True, blur = True )


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # mask  = self.file_names['img'].iloc[idx]['mask']
        # TODO: change to cv2
        image = self.load_image(idx)
        # image = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx]['img_'+self.modality[0]], mode='RGB' ))
        mask = np.array(self.PIL_loader(self.root_dir, 'masks/'+ self.file_names.iloc[idx]['mask']) )
        if self.data_augmentation:
            sample = Compose( self.data_augmentation)(image=image, mask=mask)
        return sample

    def __add__(self, other):
        return ConcatDataset([self, other])

    def split_idx(self): #TODO: find more efficient choose series
        # file_names = self.file_names.iloc[0]
        file_names = pd.DataFrame(columns=self.file_names.columns)
        for id in  self.series_id: #[0].values:
            selection = self.file_names[ self.file_names["mask"].str.contains(id) == True]
            file_names = file_names.append(selection)
        self.file_names = file_names

    def normalization(self):
        # Running Normalization of the dataset
        running_mean = 0
        running_std= 0
        for p in  self.file_names['img_'+ self.modality[0] ]:
            images = np.array(self.PIL_loader( self.root_dir , 'train/'+ p ) )
            mean, std = images.mean(), images.std()
            running_mean += mean.item()
            running_std += std.item()
        mean = running_mean / self.__len__()
        std   = running_std / self.__len__()
        print("The mean of the dataset is {0:.2f} and the standard deviation {1:.2f}".format(mean, std ))
        self.mean = mean
        self.std = std
        return  mean, std

    def save_check_data(self, **kwargs):
        """ For debugging purposes """
        result_dir = kwargs.get('result_dir', self.root_dir )
        result_dir = make_directory(result_dir, "data_processed_visualization")
        set_matplotlib_params()
        reds  = colormap_transparent(1,0,0)
        for idx in range(self.__len__()):
            # dir_path = os.path.join(result_dir, self.type + '_data_visualization', 'dataset_' + str(idx).zfill(2))
            sample = self.__getitem__(idx)
            print("Image ", idx)
            image = np.asarray(sample['image']).astype(np.float64)
            mask  = np.asarray(sample['mask']).astype(np.float64)
            fig = plt.figure(figsize=(6.40, 6.40))
            image_name =  self.file_names.iloc[idx]['mask'].split('_')
            plt.title("Image " +  image_name[2]  + " slice " +  image_name[-1][0])
            plt.imshow(image, cmap='gray', interpolation='lanczos')
            plt.imshow(mask, cmap=reds, interpolation='lanczos')
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            fig.savefig(result_dir.joinpath( str( '_').join([image_name[2]] + image_name[4:] )) )
            plt.close()
        print("The previsualization of the data is saved in folder: " + str(result_dir))
        html(result_dir, '*', '.png', title='dataset_visualization')

    @staticmethod
    def PIL_loader(path, filename="", mode = 'L'):
        with open(Path(path).joinpath(filename), 'rb') as f:
            with Image.open(f) as img:
                return img.convert( mode )

    def load_image(self,idx ):
        image = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx]['img_'+self.modality[0]], mode='RGB'))
        if len(self.modality)>1:
            for i, m in enumerate(self.modality):
                image[:,:,i] = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx]['img_'+ m ], mode='L'))
        return image

    @staticmethod
    def image_loader(path, filename):
        with open(Path(path).joinpath(filename), 'rb') as f:
            return cv2.imread(f,cv2.IMREAD_GRAYSCALE)


    def set_augmentation(self, prob = 0.5, image_size =(256,256), data_augmentation= True, **params): # **kwargs
        # Get the configuration parameters
        rotation = params.get('rotation', True)
        crop = params.get('crop', True)
        noise = params.get('noise', False)
        clahe = params.get('clahe', True)
        contrast = params.get(' contrast ', False)
        blur = params.get('blur', False)
        distorsion = params.get('distorsion', False)
        augmentation = []

        if data_augmentation and self.phase is 'train':
            if rotation:
                augmentation += OneOf([  Rotate(limit=45,interpolation=cv2.INTER_LANCZOS4 ),
                                         VerticalFlip(),
                                         HorizontalFlip(),
                                         RandomRotate90(),
                                         Transpose(),
                                         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, interpolation=cv2.INTER_LANCZOS4)
                                        ], p = prob)
            if crop:
                augmentation += OneOf([ RandomCrop( int(image_size[0]*0.875), int(image_size[1]*0.875)),
                                        CenterCrop(int(image_size[0]*0.875), int(image_size[1]*0.875) ),
                                        # RandomSizedCrop( (int(image_size[0]*0.875), int(image_size[1]*0.875)), image_size[0], image_size[1], interpolation=cv2.INTER_LANCZOS4)
                                        ], p=prob)
            if clahe:
                augmentation += [CLAHE(p=1., always_apply=True)]
            elif contrast :
                augmentation += OneOf([RandomBrightnessContrast(),   RandomGamma()], p = prob )
            if noise:
                augmentation += [GaussNoise(p=.5, var_limit=1.)]
            if distorsion:
                augmentation +=  OneOf([ GridDistortion(p=.1),
                                        ElasticTransform(p=.5, sigma=1., alpha_affine=20, border_mode=0)
                                        ], p=prob)
            if blur:
                augmentation +=  OneOf([  MotionBlur(p=.3),
                                          Blur(blur_limit=3, p=.3),
                                          GaussianBlur(blur_limit=3, p=.3)
                                        ], p=prob)

        augmentation += [Resize(image_size[0], image_size[1], cv2.INTER_LANCZOS4),
                         Normalize(mean=(self.mean), std=(self.std), max_pixel_value=255.0, always_apply=True, p=1.0),
                        ToTensor(num_classes=4)]
        return augmentation



if __name__ == "__main__":
    dataset = MyOpsDataset("./input/images_masks_full.csv", "./input", series_id= "./input/series_ID.csv")
    dataset.__getitem__(0)
    dataset.save_check_data()
    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}
    dataloader = DataLoader(dataset,**params)
