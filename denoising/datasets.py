import glob
import random
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from utils import crop_patch, random_augmentation, crop_img

class PatchSimulateDataset(Dataset):
    def __init__(self, pth, length, patch_size, sigma):
        super(PatchSimulateDataset, self).__init__()
        self.len = length
        self.patch_size = patch_size
        self.sigma = sigma

        rgb2gray = transforms.Grayscale(1)
        self.images = []
        self.num_images = 0
        for p in glob.glob(pth+'*.jpg'):
            img = Image.open(p)
            self.images.append(np.array(rgb2gray(img))[..., np.newaxis])
            self.num_images += 1

        self.transform = transforms.ToTensor()
    
    def __getitem__(self, index):
        idx = random.randint(0, self.num_images-1)
        img_patch = crop_patch(self.images[idx], self.patch_size)
        
        # generate gaussian noise N(0, sigma^2)
        noise = np.random.randn(*(img_patch.shape))
        noise_patch = np.clip(img_patch + noise*self.sigma, 0, 255).astype(np.uint8)

        aug_list = random_augmentation(img_patch, noise_patch)
        return self.transform(aug_list[1]), self.transform(aug_list[0])

    def __len__(self):
        return self.len