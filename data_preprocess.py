'''
Data Preprocessing and Preperation
'''

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random
import torch 
import torchvision.transforms as T 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
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
        #print(self.target_files[index])
        print(index)
        image = Image.open(self.image_paths + self.image_files[index])
        mask = Image.open(self.target_paths + self.target_files[index])
        x, y = self.transform(image, mask)
        return x, y
        

    def __len__(self):
        return len(self.image_files)
    

def save_transData(in_dir_im, in_dir_mask, out_dir_im, out_dir_mask):
    dataset_list = []
    mydataset = MyDataset(in_dir_im, in_dir_mask)
    for i in range(len(os.listdir(in_dir_im))):
        dataset_list.append(mydataset.__getitem__(i))

    for i, data in enumerate(dataset_list):
        torch.save(data[0], os.path.join(out_dir_im, 'trans_im_{}'.format(i)))
        torch.save(data[1], os.path.join(out_dir_mask, 'trans_mask_{}'.format(i)))
    

class transformed_data(Dataset):
  def __init__(self, img, mask):
    self.img = img  #img path
    self.mask = mask  #mask path
    self.len = len(os.listdir(self.img))

  def __getitem__(self, index):
    ls_img = sorted(os.listdir(self.img))
    ls_mask = sorted(os.listdir(self.mask))

    img_file_path = os.path.join(self.img, ls_img[index])
    img_tensor = torch.load(img_file_path)
    
    mask_file_path = os.path.join(self.mask, ls_mask[index])
    mask_tensor = torch.load(mask_file_path)

    return img_tensor, mask_tensor

  def __len__(self):
    return self.len   

    


if __name__ == '__main__':
    in_dir_im, in_dir_mask = 'data/inputs_small/', 'data/semantic_annotations_small/'
    #out_dir_im, out_dir_mask = 'data/inputs_transformed/', 'data/semantic_annotations_transformed/'


    data_set = MyDataset(in_dir_im, in_dir_mask)
    data_loader = DataLoader(data_set, batch_size=5, shuffle=True)

    for (i, (x, y)) in enumerate(data_loader):
        print(x.shape)
        print(y.shape)

    # save_transData(in_dir_im, in_dir_mask, out_dir_im, out_dir_mask)
    
    # ds = transformed_data(out_dir_im, out_dir_mask)

    # x, y = ds.__getitem__(0)
    # x = torch.sum(input=x, dim=0)
    
    # im_mat = x 
    # an_mat = y

    # fig, ax = plt.subplots(1,2, figsize=(13,15))

    # ax[0].set_title('Semantic Annotations')
    # ax[1].set_title('Original Image')


    # ax[0].imshow(an_mat, cmap='brg', vmin=1, vmax=17)
    # ax[1].imshow(im_mat, cmap='gray')

    # plt.show()


