"""
@Author: Sulaiman Vesal, Maria Monzon
"""

import torch
import pandas as pd
# from skimage.exposure import match_histograms
from utils.image_proces_vis import *
from matplotlib import pyplot as plt
import nrrd
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
from albumentations.pytorch.transforms import ToTensor
from utils.save_data import html, make_directory
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from utils.utils import one_hot2dist
from PIL import Image
from pathlib import Path

class MyOpsDataset(Dataset):
    MODALITIES = {'CO': 0, 'DE': 1, 'T2': 2}
    def __init__(self, csv_path, root_path, transform = False, series_id = "", split = True, phase ='train', image_size = (256, 256), n_classes = 6, modality = ['CO', 'DE', 'T2']):
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
        self.modality =  self.set_modality( modality)#modality if isinstance(modality, list) else [modality]
        if split and series_id is not "":
            self.series_id = list(series_id)
            self.file_names = self.split_idx()
        if Path(root_path).is_absolute():
            self.root_dir =  Path(root_path)
        else:
            self.root_dir = Path(__file__).resolve().parent.joinpath(root_path)
        self.mean, self.std = self.normalization()
        self.image_size = image_size
        self.phase = phase
        self.num_classes =  n_classes
        self.data_augmentation = self.set_augmentation( 0.5, image_size, data_augmentation = transform,  rotation= True,
                                                        crop = True, clahe=True, noise=True, blur = True )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        # image = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx]['img_'+self.modality[0]], mode='RGB' ))
        mask = np.array(self.PIL_loader(self.root_dir, 'masks/'+ self.file_names.iloc[idx]['mask']) )
        # HISTOGRAM EQUALIZATION

        if self.data_augmentation:
            sample = Compose( self.data_augmentation)(image=image, mask=mask)

        # Get mask with as one-hot mask in each channel [1, n_clases, image_size[0], image_size[1])
        if self.num_classes ==1:
            sample['mask'] = self.binary_mask(sample['mask'])
        else:
            sample['mask'] = self.categorical_maks( sample['mask'] )
        # Get distance maps for Boundary loss
        sample['distance_map'] =self.distance_map(sample['mask'])

        return sample

    def __add__(self, other):
        return ConcatDataset([self, other])

    def split_idx(self): #TODO: find more efficient implementation
        file_names = pd.DataFrame(columns=self.file_names.columns)
        for id in  self.series_id:
            selection = self.file_names[ self.file_names["mask"].str.contains(id) == True]
            file_names = file_names.append(selection)
        return file_names

    def normalization(self):
        # Running Normalization of the dataset
        running_mean = 0
        running_std= 0
        for m in range(len(self.modality)):
            k = self.file_names.columns[0]
            for p in  self.file_names[k]:
                images = np.array(self.PIL_loader( self.root_dir , 'train/'+ p ) )
                mean, std = images.mean(), images.std()
                running_mean += mean.item()
                running_std += std.item()
        mean = running_mean / (len(self.modality)*self.__len__() )
        std   = running_std / (len(self.modality)*self.__len__() )
        # print("The mean of the dataset is {0:.2f} and the standard deviation {1:.2f}".format(mean, std ))
        self.mean = mean
        self.std = std
        return  mean, std

    def set_modality(self, modality):
        # elper function to choose the image modality (CO, DE, T2)
        mod = []
        for m in modality:
            mod.append(int(self.MODALITIES[m]))
        print("The image modalities are: ", mod)
        return mod

    def set_augmentation(self, prob = 0.5, image_size =(256,256), data_augmentation= True, **params): # **kwargs
        # Get the configuration parameters for augmentation
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
                augmentation += OneOf([  Rotate(limit=45, interpolation=cv2.INTER_LANCZOS4 ),
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
                         # Normalize(mean=(self.mean), std=(self.std), max_pixel_value=255.0, always_apply=True, p=1.0),
                         ToTensor(num_classes=self.num_classes-1)]
        return augmentation

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
        key = self.file_names.columns[self.modality[0]]
        image = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx][key], mode='RGB'))
        if len(self.modality)>1:
            # key = self.file_names.columns[0]
            for i, m in enumerate(self.modality):
                key = self.file_names.columns[m]        # key ='img_' + m
                image[:,:, i] = np.array(self.PIL_loader( self.root_dir , 'train/'+ self.file_names.iloc[idx][ key ], mode='L'))
        return image

    def categorical_maks(self, mask):
        """Converts a class vector to binary class matrix."""
        num_classes =  self.num_classes #len(torch.unique(mask, sorted=True)) #  np.unique(mask )
        # assert num_classes > len(torch.unique(mask))
        channel_mask = torch.zeros((num_classes,) + mask.shape)
        for n, c in enumerate(torch.unique(mask, sorted=True)):
            channel_mask[n] = (mask == c)
        return channel_mask.float()

    def binary_mask(self, mask):
        return (mask > 0.).float()

    @staticmethod
    def image_loader(path, filename):
        with open(Path(path).joinpath(filename), 'rb') as f:
            return cv2.imread(f,cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def distance_map(mask):
        maps = one_hot2dist(mask.cpu().numpy(), axis =0)
        return  torch.from_numpy(maps)


def extract_nrrd_data(PATH=r"/human-dataset", CLAHE = False):
    # import nrrd
    from glob import  iglob
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    for p in iglob(PATH+str('/*de.nrrd')):
        id = 126
        dir_path =Path( str(p).split('_')[0])
        # Create Directory
        dir_path.mkdir(parents=True, exist_ok = True)
        images = nrrd.read(p)[0]
        for i in range(images.shape[-1]):
            img = images[:,:,i]
            img = (255 * (img- img.min()) / (img.max() - img.min())).astype(np.uint8)
            if CLAHE:
                img = clahe.apply(img)
            img = Image.fromarray(img, mode="L")
            img.save(dir_path.joinpath('myops_training_{0}_DE_{1}.png'.format(id, str(i).zfill(2))))
        id+=1

if __name__ == "__main__":
    from glob import  iglob
    # PATH=r"D:\OneDrive - fau.de\1.Medizintechnik\5SS Praktikum\human-dataset"
    # extract_nrrd_data(PATH=PATH)
    dataset = MyOpsDataset("./input/images_masks_full.csv", "./input", series_id= np.arange(101,110).astype(str), n_classes=6, modality = ['T2', 'DE'])
    for idx in range(5)
        sample = dataset.__getitem__(idx)
        mask = sample['mask']
        image= sample['image']
        plt.imshow(image[0])
        plt.figure()
        for i in range(3 * 2):
            plt.subplot(2, 3, i + 1)
            plt.imshow(mask[i])
            plt.axis('off')

    dataset.save_check_data()
    params = {'batch_size': 16, 'shuffle': True, 'num_workers': 6}
    dataloader = DataLoader(dataset,**params)
