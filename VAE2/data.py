import os
import numpy as np
import PIL.Image

from torch.utils.data import Dataset, DataLoader


class RealPhotosDataset(Dataset):
    '''
    Generate the dataset used in the VAE2 model

    Args:
        directory (str): Path to the directory containing the ground truth images

    Attributes:
        files_r (list): List containing all the ground truth images
    '''

    def __init__(self, directory):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory)]

    def __getitem__(self, index):
        img = PIL.Image.open(self.files[index]).convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        img = np.transpose(img,(2,0,1)).astype('float32')
        return torch.tensor(img-0.5)
    
    def __len__(self):
      return len(self.files)
