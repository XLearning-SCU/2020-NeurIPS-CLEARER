import glob
import random
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from utils import random_augmentation

# Rain800 dataset
class Rain800(Dataset):
    def __init__(self, pth, length, patch_size):
        super(Rain800, self).__init__()
        self.rain_images = []
        self.clear_images = []
        self.patch_size = patch_size
        self.len = length
        self.num_images = 0
        for p in glob.glob(pth+'*.jpg'):
            img = np.array(Image.open(p))
            assert img.shape[1]%2==0
            self.clear_images.append(img[:,:img.shape[1]//2,:])
            self.rain_images.append(img[:,img.shape[1]//2:,:])
            self.num_images += 1
            # Image.fromarray(img[:,:img.shape[1]//2,:]).save('datasets/rain800/training_split/clear/'+p.split('/')[-1])
            # Image.fromarray(img[:,img.shape[1]//2:,:]).save('datasets/rain800/training_split/rain/'+p.split('/')[-1])
        self.transform = transforms.ToTensor()

    def __getitem__(self,index):
        idx = random.randint(0, self.num_images-1)
        H = self.clear_images[idx].shape[0]
        W = self.clear_images[idx].shape[1]
        ind_H = random.randint(0, H-self.patch_size)
        ind_W = random.randint(0, W-self.patch_size)

        clear_patch = self.clear_images[idx][ind_H:ind_H+self.patch_size, ind_W:ind_W+self.patch_size]
        rain_patch = self.rain_images[idx][ind_H:ind_H+self.patch_size, ind_W:ind_W+self.patch_size]
        aug_list = random_augmentation(clear_patch, rain_patch)
        return self.transform(aug_list[1]), self.transform(aug_list[0])

    def __len__(self):
        return self.len