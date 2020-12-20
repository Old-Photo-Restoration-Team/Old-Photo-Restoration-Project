import os
import numpy as np
import PIL.Image



class CreateDataSet(Dataset):
    def __init__(self, path_r,path_x):
        self.files_r = [os.path.join(path_r, f) for f in os.listdir(path_r)]
        self.files_x = [os.path.join(path_x, f) for f in os.listdir(path_x)]

    def __getitem__(self, index):
        img_r=self.Read_image(self.files_r[index]);
        img_x=self.Read_image(self.files_x[index]);
        return img_r,img_x

    def Read_image(self,name):
        img = PIL.Image.open(name).convert('RGB')

        return preprocessing(img)

    def __len__(self):
      return min([len(self.files_r),len(self.files_x)])