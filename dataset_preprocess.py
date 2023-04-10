
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random
import torch 
import torchvision.transforms as T 
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader




class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_files = os.listdir(image_paths)
        self.target_files = os.listdir(target_paths)
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.train = train

    def transform(self, image, mask):

        if self.train:
            # Random crop
            if random.random() > 0.5:
                i, j, h, w = T.RandomCrop.get_params(
                    image, output_size=(2000, 1300))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            

            # Resize
            resize = T.Resize(size=(2200, 1550))
            image = resize(image)
            mask = resize(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return image, mask
    


    def __getitem__(self, index):
        image = Image.open(self.image_paths + self.image_files[index])
        mask = Image.open(self.target_paths + self.target_files[index])
        x, y = self.transform(image, mask)
        return x, y
        

    def __len__(self):
        return len(self.image_files)
    


if __name__ == '__main__':
    in_dir_im, in_dir_mask = 'data/inputs_small/', 'data/semantic_annotations_small/'

    data_set = MyDataset(in_dir_im, in_dir_mask, train=True)
    data_loader = DataLoader(data_set, batch_size=5, shuffle=True)

    num_iters = 6
    # iterate through 6 passes of the dataset
    for _ in range(num_iters):
        # iterate through dataset and augment images
        for (i, (x, y)) in enumerate(data_loader):
            print(x.shape)