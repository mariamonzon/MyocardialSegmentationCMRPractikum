"""
@Author: Sulaiman Vesal, Maria Monzon
"""

import torch
from torch import Tensor
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
from skimage.measure import label, regionprops
from utils.save_data import html, make_directory
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from utils.utils import one_hot2dist
from PIL import Image
from pathlib import Path


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




class MyOpsDataset(Dataset):
    MODALITIES = {'CO': int(0), 'DE': int(1), 'T2': int(2)}
    def __init__(self, csv_path, root_path, augmentation = False, series_id ="", split = True, phase ='train', image_size = (256, 256), n_classes = 6, modality = ['CO', 'DE', 'T2'], crop_center=0):
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
        # self.mean, self.std = self.normalization()
        self.image_size = image_size
        self.phase = phase
        self.num_classes =  n_classes
        self.data_augmentation = augmentation
        transforms_dict = {'rotation': True, 'crop' : True, 'clahe': True, 'noise': True, 'blur': True}
        self.transforms = self.set_transforms(0.5, image_size, data_augmentation = augmentation, **transforms_dict)
        self.crop = int(crop_center/2)
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        # mask = np.array(self.PIL_loader(self.root_dir, 'masks/'+ self.file_names.iloc[idx]['mask']) )
        mask  =  np.load( Path(self.root_dir).joinpath('masks/'+self.file_names.iloc[idx]['mask']) )

        if self.crop: # and self.phase == 'train':
            center = self.find_center(self.binary_mask(mask)[1],  label= 1)
            image = self.crop_images(image.copy(), center, window=0.25*max(image.shape))
            mask = self.crop_images(mask,center,window=0.25*max(mask.shape))

        # HISTOGRAM EQUALIZATION
        sample = Compose( self.transforms)(image=image, mask=mask)

        # Get mask with as one-hot mask in each channel [n_clases, image_size[0], image_size[1])
        if self.num_classes ==1 or self.num_classes == 2 :
            sample['mask'] = self.binary_mask(sample['mask'])
            # sample['mask'] = self.single_class_mask(sample['mask'])
            # sample['mask'] = self.categorical_merge_labels( sample['mask'], labels = [1, 4, 5])
        elif self.num_classes == 3:
            sample['mask'] = self.categorical_maks_labels(sample['mask'])
        elif self.num_classes == 4:
            sample['mask'] = self.categorical_maks_labels( sample['mask'], labels= [1,2,4])
        elif self.num_classes == 5:
            sample['mask'] = self.categorical_maks_labels(sample['mask'], labels=[1,2,4,5])
        else:
            sample['mask'] = self.categorical_maks( sample['mask'] )

        # Get distance maps for Boundary loss
        sample['distance_map'] = self.distance_map(sample['mask'])
        sample = self.ToTensor(sample)
        return  sample

    def __add__(self, other):
        return ConcatDataset([self, other])

    def split_idx(self): #TODO: find more efficient implementation
        file_names = pd.DataFrame(columns=self.file_names.columns)
        for id in  self.series_id:
            selection = self.file_names[ self.file_names["mask"].str.contains(id) == True]
            file_names = file_names.append(selection)
        if self.modality == [] or self.modality == ['multi']:
            file_names = file_names.melt(id_vars=["mask"])
            file_names = file_names.rename(columns={'value': 'image', 'variable': 'modality'})
            self.modality = [-1]
        return file_names

    def normalization(self):
        # Running Normalization of the dataset
        running_mean = 0
        running_std= 0
        for m in self.modality:
            k = self.file_names.columns[m]
            for p in self.file_names[k]:
                images = np.array(self.PIL_loader( self.root_dir , 'train/'+ p ) )
                mean, std = images.mean(), images.std()
                running_mean += mean.item()
                running_std += std.item()
        mean = running_mean / (len(self.modality)*len(self.file_names))
        std   = running_std / (len(self.modality)*len(self.file_names) )
        # print("The mean of the dataset is {0:.2f} and the standard deviation {1:.2f}".format(mean, std ))
        self.mean = mean
        self.std = std
        return  mean, std

    def set_modality(self, modality):
        # elper function to choose the image modality (CO, DE, T2)
        mod = []
        if modality != ['multi']:
            for m in modality:
                mod.append(self.MODALITIES.get(m, 0))
        return mod
    @property
    def filename(self, idx):
        return self.file_names[idx]['mask']

    @property
    def augmentation(self):
        return self.data_augmentation

    def set_augmentation(self, augmentation,  params = {'rotation': True, 'crop': True, 'clahe': True, 'noise': True, 'blur': True, 'distorsion':True}, prob =0.95):
        assert isinstance(augmentation, bool), "augmentation has to be bool"
        self.data_augmentation = augmentation
        self.transforms = self.set_transforms(prob, self.image_size, data_augmentation = augmentation, **params)

    def set_transforms(self, prob = 0.5, image_size =(256, 256), data_augmentation= True, **params): # **kwargs
        # Get the configuration parameters for augmentation
        rotation = params.get('rotation', True)
        crop = params.get('crop', True)
        noise = params.get('noise', False)
        clahe = params.get('clahe', True)
        contrast = params.get(' contrast ', True)
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
            if contrast :
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
            if clahe:
                augmentation += [CLAHE(p=1., always_apply=True)]
        augmentation += [Resize(image_size[0], image_size[1], cv2.INTER_LANCZOS4)]
                         # Normalize(mean=(self.mean), std=(self.std), max_pixel_value=255.0, always_apply=True, p=1.0),
                         # ToTensor(num_classes=self.num_classes-1)]
        return augmentation


    def save_check_data(self, **kwargs):
        """ For debugging purposes """
        from utils.utils import categorical_mask2image
        result_dir = kwargs.get('result_dir', self.root_dir )
        result_dir = make_directory(result_dir, "data_processed_visualization")
        set_matplotlib_params()
        reds  = colormap_transparent(1,0.5,0)
        for idx in range(self.__len__()):
            # dir_path = os.path.join(result_dir, self.type + '_data_visualization', 'dataset_' + str(idx).zfill(2))
            sample = self.__getitem__(idx)
            print("Image ", idx)
            image = sample['image'].cpu().numpy( ).astype(np.float64)
            mask = np.zeros(self.image_size)
            for c in range(sample['mask'].shape[0]):
                mask += 255*c*sample['mask'][c].cpu().numpy( )
            fig = plt.figure(figsize=(6.40, 6.40))
            image_name =  self.file_names.iloc[idx%len(self.file_names)]['mask'][:-4].split('_')
            plt.title("Image " +  image_name[2]  + " slice " +  image_name[-1][0])
            plt.imshow(image[0], cmap='gray', interpolation='lanczos')
            plt.imshow(mask , cmap=reds, interpolation='lanczos')
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            filename = 'Image_{}_'.format(idx) + str( '_').join([image_name[2]] + image_name[4:])+'.png'
            fig.savefig(result_dir.joinpath( filename))
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

    def categorical_maks_torch(self, mask: Tensor):
        """Converts a class vector to binary class matrix."""
        num_classes =  self.num_classes #len(torch.unique(mask, sorted=True)) #  np.unique(mask )
        # assert num_classes > len(torch.unique(mask))
        channel_mask = torch.zeros((num_classes,) + mask.shape)
        for n, c in zip(range(num_classes) , torch.unique(mask, sorted=True)):
            channel_mask[n] = (mask == c)
        return channel_mask.float()

    def categorical_maks(self, mask: np.ndarray):
        """Converts a class vector to binary class matrix."""
        num_classes =  self.num_classes
        channel_mask = np.zeros((num_classes,) + mask.shape).astype(np.float32)
        for n, c in zip(range(num_classes) , np.unique(mask)):
            channel_mask[n] = (mask == c)
        return channel_mask

    def categorical_maks_labels(self, mask: np.ndarray, labels =[4,5] ):
        """Converts a class vector to binary class matrix."""
        num_classes =  self.num_classes
        channel_mask = np.zeros((num_classes,) + mask.shape).astype(np.float32)
        for n, c in zip(range(1, num_classes), labels):
            channel_mask[n] = (mask == c)
        # Set the rest of the classes as background
        channel_mask[0] = (channel_mask !=  1).astype(float).sum(axis=0)

        return channel_mask

    def categorical_merge_labels(self, mask: np.ndarray, labels=[2, 4, 5]):
        """Converts a class vector to binary class matrix."""
        num_classes = self.num_classes
        channel_mask = np.zeros((num_classes,) + mask.shape).astype(np.float32)
        channel_mask[-1] =  (mask ==  labels[0]) | (mask ==  labels[1])
        #for  c in zip(range(1, num_classes-1)):
        #   channel_mask[c] = (mask == c)
        # Set the rest of the classes as background
        channel_mask[0] = (channel_mask != 1).copy().astype(float).sum(axis=0)

        return channel_mask

    def single_class_mask(self, mask, label=4.):
        bin_mask =  np.zeros((2,) + mask.shape).astype(np.float32)
        bin_mask[0] =  (mask !=  label)
        bin_mask[1] =  (mask ==  label)
        return (bin_mask).astype(float)

    def binary_mask(self, mask, label=3.):
        bin_mask =  np.zeros((2,) + mask.shape).astype(np.float32)
        # rv = 68
        bin_mask[0] =  (mask ==  label) | (mask == 0.)
        # bin_mask[1] = ~bin_mask[0]
        bin_mask[1] =  (mask !=  label) & (mask > 0.)
        return (bin_mask).astype(float)

    def ToTensor(self, sample):
        return {'image': torch.from_numpy(np.rollaxis(sample['image'], -1, 0) /255.).float(), 'mask': torch.from_numpy(sample['mask']).float(), 'distance_map': torch.from_numpy(sample['distance_map']/255.).float()}

    @staticmethod
    def image_loader(path, filename):
        with open(Path(path).joinpath(filename), 'rb') as f:
            return cv2.imread(f,cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def distance_map(mask):
        maps = one_hot2dist(mask, axis =0)
        return  maps # torch.from_numpy(maps) #



    # def image_crop(self, mask, c = 255, window = 64):
    #
    #
    #     return image
    def crop_gt(self):
        mod = ['C0', 'DE', 'T2']
        for idx in range(len(self.file_names)):
            image = self.load_image(idx)
            mask_path = Path(self.root_dir).joinpath('masks/' + self.file_names.iloc[idx]['mask'])
            original_mask = np.load(mask_path)
            bin_mask = self.binary_mask(original_mask )
            regions = regionprops(bin_mask[1].astype(int))
            #center = regions[0].centroid
            bbox = regions[0].bbox
            dir_data = make_directory(self.root_dir, 'train_crop_bbox/')
            dir_mask = make_directory(self.root_dir, 'masks_crop_bbox/')
            image = self.crop_bbox(image, bbox, pad=5)
            original_mask = self.crop_bbox(original_mask, bbox, pad=5)

            for i, m in enumerate(self.modality):
                key = self.file_names.columns[i]  # key ='img_' + m
                Image.fromarray(image[:, :, i]).save(dir_data.joinpath(self.file_names.iloc[idx][key]))
                mask_name = str( self.file_names.iloc[idx]['mask']).replace('C0', mod[m])
                np.save(dir_mask.joinpath(mask_name), original_mask.astype(np.uint8))



    def save_crop_images(self, mask, idx):
        image = self.load_image(idx)
        mask = cv2.resize(mask[0], image.shape[:2])
        mask_path = Path(self.root_dir).joinpath('masks/' + self.file_names.iloc[idx]['mask'])
        original_mask = np.load(mask_path)
        regions = regionprops(mask.astype(int))
        if len(regions)==0:
            center =  regionprops(original_mask.astype(int))[0].centroid
        else:
            center = regions[0].centroid
        #  bbox = regions[0].bbox
        dir_data = make_directory(self.root_dir , 'train_crop_128_128/')
        dir_mask = make_directory(self.root_dir, 'masks_crop_128_128/')

        image = self.crop_images(image,center, window=64)
        original_mask = self.crop_images(original_mask, center, window=64)

        # original_mask = self.crop_bbox(original_mask, bbox, pad=10)
        # image = self.crop_bbox(image, bbox, pad=10)
        # image = Image.fromarray(image).resize((96,96))
        # image  = np.fromstring(image.tobytes(), dtype=np.uint8).reshape((image.size[1], image.size[0], 3))
        # image = cv2.resize(image, (96,96), interpolation=0)
        # original_mask = cv2.resize(original_mask,(96,96), interpolation=0)

        for i, m in enumerate(self.modality):
            key = self.file_names.columns[m]        # key ='img_' + m
            # cv2.imwrite(  Path(self.root_dir /'train_crop'/ self.file_names.iloc[idx][key]) , image[:,:,i])
            # image[:, :, i].save(dir_data.joinpath(self.file_names.iloc[idx][key]))
            Image.fromarray(image[:, :, i]).save(dir_data.joinpath( self.file_names.iloc[idx][key]) )
        np.save( dir_mask.joinpath( self.file_names.iloc[idx]['mask']) ,  original_mask.astype(np.uint8))

        del image, mask,   original_mask, dir_data, dir_mask
        return None

    @staticmethod
    def find_center(mask, label = 1):
        (y, x) = np.where(mask == label)
        cx = int(np.mean(x))
        cy = int(np.mean(y))
        return cy, cx

    @staticmethod
    def crop_bbox(images, bbox, pad=0):
        bb01, bb00, bb11, bb10 = bbox
        # print(bb01,bb11, bb00,bb10 )
        images = images[ -pad+bb01:bb11+pad, -pad+bb00:bb10+pad ]
        return images


    @staticmethod
    def crop_images( image, center, window=128/2):

        txl = max(int(center[0] - window), 0)
        tyl = max(int(center[1] - window), 0)

        txr = min(int(center[0] + window), int(image.shape[0]))
        tyr = min(int(center[1] + window), int(image.shape[1]))

        image = image[txl:txr , tyl:tyr ]
        return image


    @staticmethod
    def create_csv( folder_path, mask_path):
        modalities  = ['C0', 'DE', 'T2']
        files = pd.DataFrame(columns=modalities+['mask'])
        for m in modalities:
            files[str(m)]=  [p.relative_to(folder_path) for p in Path(folder_path).glob('*_'+m+'_*.png')]
        files['mask'] = [pm.relative_to(mask_path) for pm in Path(mask_path).glob('*_'+'C0'+'_*.npy')]
        files.to_csv('./input/filenames.csv', sep=';')



class MyOpsDatasetHeatmaps(MyOpsDataset):
    def __init__(self, csv_path, root_path, augmentation=False, series_id="", split=True, phase='train',
                 image_size=(256, 256), n_classes=1, modality=['CO', 'DE', 'T2'], sd=16):
        super().__init__(csv_path, root_path, augmentation, series_id, split, phase, image_size, n_classes, modality)
        self.sigma = sd
        if augmentation:
            super().set_augmentation(True)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = np.load(Path(self.root_dir).joinpath('masks/' + self.file_names.iloc[idx]['mask']))
        sample = Compose(self.transforms)(image=image, mask=mask)
        sample['mask'] = self.binary_mask(sample['mask'])
        center =  regionprops(sample['mask'][1].astype(int))[0].centroid # self.find_center(self.binary_mask(mask)[1], label=1)
        sample['heatmaps'] = self.gaussian2d( center,(self.sigma,self.sigma), self.image_size, amplitude= 1, offset=0, normalize=False)
        sample['landmark'] = center[::-1]
        # Get distance maps for Boundary loss
        sample = self.ToTensor(sample)
        return sample

    def sigma_reduce(self, s=4):
        self.sigma -= s

    @staticmethod
    def gaussian2d( centroid, sigma=(32,32), image_size=(256,256), amplitude = 1, offset=0, normalize=False):
        r"""Create the Net2D Gaussian for the given coordinates by pos_tupel
        math::
        \begin{equation}
            G(x,y) = A \exp\left(- \left(a(x - x_o)^2 + 2b(x-x_o)(y-y_o) + c(y-y_o)^2 \right)\right) + Offset
        \end{equation}
        """
        # Simple Gaussian Coefficients
        # (x, y) = xy_grid
        (xo, yo) = centroid
        (sigma_x, sigma_y) = sigma
        x,y = torch.meshgrid(torch.arange(0, image_size[0]).float() , torch.arange(0, image_size[1]).float())
        x_fac = 1/ (2 * sigma_x ** 2)
        y_fac = 1 / (2 * sigma_y ** 2)

        # G = O + A * e ^ -[(x-xo)^2/(2*sig_y^2)  + (y-yo)^2/(2*sig_y^2) ]
        g = offset + amplitude * torch.exp(- (x_fac * ((x - xo) ** 2)  + y_fac * ((y - yo) ** 2)))

        if normalize:        #Enforce to be a pdf
            g = g / torch.sum(g)
        return g


    def ToTensor(self, sample):
        return {'image': torch.from_numpy(np.rollaxis(sample['image'], -1, 0) /255.).float(),'landmark':torch.Tensor(sample['landmark']).float(), 'heatmaps': torch.unsqueeze(sample['heatmaps'].float(), dim=0)}


class MyOpsDatasetAugmentation(MyOpsDataset):
    def __init__(self, csv_path, root_path, augmentation = False, series_id ="", split = True, phase ='train', image_size = (256, 256), n_classes = 6, modality = ['CO', 'DE', 'T2'], n_samples = 500):
        super(MyOpsDatasetAugmentation, self).__init__( csv_path, root_path, augmentation , series_id, split, phase, image_size, n_classes, modality)
        self.n_samples = n_samples if n_samples != -1 else len(self.file_names)
        self.sample = self.n_samples *  [None]
        # idx = np.arange(n_samples)
        for i in np.arange(self.n_samples):
            self.sample[i] = super().__getitem__(i % len(self.file_names))
            if i % len(self.file_names)+1 == 0:
                super().set_augmentation(True)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.sample[idx]



class MyOpsDatasetPatches(MyOpsDataset):
    def __init__(self, csv_path, root_path, augmentation = False, series_id ="", split = True, phase ='train', image_size = (128, 128), n_classes = 6, modality = ['CO', 'DE', 'T2'], n_samples = 500):
        super().__init__( csv_path, root_path, augmentation , series_id, split, phase, image_size, n_classes, modality)
        self.patch_size=32
        self.stride =  16 if phase is 'train' else self.patch_size

    def __getitem__(self, idx):

        image = self.load_image(idx)
        # mask = np.array(self.PIL_loader(self.root_dir, 'masks/'+ self.file_names.iloc[idx]['mask']) )
        mask = np.load(Path(self.root_dir).joinpath('masks/' + self.file_names.iloc[idx]['mask']))

        # HISTOGRAM EQUALIZATION
        sample = Compose(self.transforms)(image=image, mask=mask)

        sample['scar'] = self.categorical_maks_labels( sample['mask'], labels=[4])
        sample['edema'] = self.categorical_merge_labels(sample['mask'], labels=[4,5])
        sample = self.ToTensor(sample)

        image = sample['image']
        image = image.unfold(-2, self.patch_size, self.stride).unfold(-1,self.patch_size , self.stride)
        image = image.contiguous().view(image.size(0), -1, self.patch_size,self.patch_size)
        mask = sample['scar']
        mask = mask.unfold(-2, self.patch_size, self.stride).unfold(-1, self.patch_size, self.stride)
        mask = mask.contiguous().view(mask.size(0), -1, self.patch_size, self.patch_size)
        mask2 = sample['edema']
        mask2 = mask2.unfold(-2, self.patch_size, self.stride).unfold(-1, self.patch_size, self.stride)
        mask2 = mask2.contiguous().view(mask.size(0), -1, self.patch_size, self.patch_size)

        sample =  {'image': image.transpose(0,1), 'scar':mask.transpose(0,1),'edema':mask2.transpose(0,1)}
        return sample

    def ToTensor(self, sample):
        return {'image': torch.from_numpy(np.rollaxis(sample['image'], -1, 0) /255.).float(), 'scar': torch.from_numpy(sample['scar']).float(), 'edema': torch.from_numpy(sample['edema']).float()}


class Test(MyOpsDataset):
    def __init__(self, csv_path, root_path,  image_size=(128, 128), n_classes=6, modality=['CO', 'DE', 'T2']):
        super().__init__(csv_path, root_path, False, "",  False, "test",  image_size, n_classes, modality)

    def load_image(self,idx ):
        key = self.file_names.columns[self.modality[0]]
        image = np.array(self.PIL_loader( self.root_dir , self.file_names.iloc[idx][key], mode='RGB'))
        if len(self.modality)>1:
            # key = self.file_names.columns[0]
            for i, m in enumerate(self.modality):
                key = self.file_names.columns[m]        # key ='img_' + m
                image[:,:, i] = np.array(self.PIL_loader( self.root_dir , self.file_names.iloc[idx][ key ], mode='L'))
        return image

    def __getitem__(self, idx):
        image = self.load_image(idx)
        image = Compose(self.transforms)(image = image)
        return torch.from_numpy(np.rollaxis(image['image'], -1, 0) /255.).float()


    def save_crop_images(self, mask, idx):
        image = self.load_image(idx)

        mask = cv2.resize(mask, image.shape[:2])
        regions = regionprops(mask.astype(int))
        # center = regions[0].centroid
        bbox = regions[0].bbox
        dir_data = make_directory(self.root_dir , 'test_crop_bbox/')
        image = self.crop_bbox(image, bbox, pad=5)# self.crop_images(image,center, window=64)

        for i, m in enumerate(self.modality):
            key = self.file_names.columns[m]        # key ='img_' + m
            Image.fromarray(image[:, :, i]).save(dir_data.joinpath( self.file_names.iloc[idx][key]) )


class MyOpsDatasetAugmentationPatch(MyOpsDatasetPatches):
    def __init__(self, csv_path, root_path, augmentation = False, series_id ="", split = True, phase ='train', image_size = (128, 128), n_classes = 6, modality = ['CO', 'DE', 'T2'], n_samples = 500):
        super(MyOpsDatasetAugmentationPatch, self).__init__( csv_path, root_path, augmentation , series_id, split, phase, image_size, n_classes, modality)
        self.n_samples = n_samples if n_samples != -1 else len(self.file_names)
        self.sample = self.n_samples *  [None]
        # idx = np.arange(n_samples)
        for i in np.arange(self.n_samples):
            self.sample[i] = super().__getitem__(i % len(self.file_names))
            if i % len(self.file_names)+1 == 0:
                super().set_augmentation(True)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.sample[idx]


if __name__ == "__main__":
    #folder_path = r"./input/test_orig"
    #modalities = ['C0', 'DE', 'T2']
    #files = pd.DataFrame(columns=modalities + ['mask'])
    #for m in modalities:
    #    files[str(m)] = [p.relative_to(folder_path) for p in Path(folder_path).glob('*_' + m + '_*.png')]
    #    files.to_csv('./input/test_orig/test_filenames.csv', sep=';')
    #exit(0)
    from glob import  iglob
    n_clas = 1
    # PATH=r"D:\OneDrive - fau.de\1.Medizintechnik\5SS Praktikum\human-dataset"
    # extract_nrrd_data(PATH=PATH)
    # dataset = MyOpsDatasetAugmentation("./input/images_masks_modalities.csv", "./input", series_id=np.arange(101, 110).astype(str),  n_classes= n_clas , modality=['CO', 'DE', 'T2'],n_samples =500)
    dataset = MyOpsDataset("./input/resampled_input/filenames.csv", "./input/resampled_input", series_id= np.arange(101,126).astype(str), n_classes= 5,  modality=['CO', 'DE', 'T2'])
    dataset.crop_gt()
    exit(0)
    for idx in range(15, 20):
        sample = dataset.__getitem__(idx)
        mask = sample['heatmaps']
        image= sample['image']
        plt.figure()
        plt.imshow(image[0],  cmap='gray')
        plt.scatter(sample['landmark'][0],sample['landmark'][1])
        plt.figure()
        for i in range( n_clas):
            plt.subplot(2,  int(2*n_clas/4+1), i + 1)
            plt.imshow(mask[i])
            plt.scatter(sample['landmark'][0], sample['landmark'][1])
            plt.axis('off')
    plt.show()
    #
    # dataset.save_check_data()
    params = {'batch_size': 16, 'shuffle': True, 'num_workers': 6}
    dataloader = DataLoader(dataset,**params)
