import os
import numpy as np
import PIL.Image

from torch.utils.data import Dataset, DataLoader


class FakeAndRealOldPhotosGenerator(Dataset):
    '''
    Generate the dataset used in the VAE2 model  

    Args:
        path_r (str): Path to the directory containing the real old photos
        path_x (str): Path to the directory containing the fake old photos

    Attributes:
        files_r (list): List containing all the real old photos
        files_x (list): List containing all the fake old photos
    '''
    
    def __init__(self, path_r, path_x):
        self.files_r = [os.path.join(path_r, f) for f in os.listdir(path_r)]
        self.files_x = [os.path.join(path_x, f) for f in os.listdir(path_x)]

    def __getitem__(self, index):
        img_r = self.read_image(self.files_r[index])
        img_x = self.read_image(self.files_x[index])
        return img_r, img_x

    def read_image(self, name):
        img = PIL.Image.open(name).convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        img = np.transpose(img, (2,0,1)).astype('float32')
        return torch.tensor(img-0.5)

    def __len__(self):
      return min([len(self.files_r), len(self.files_x)])