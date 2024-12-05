import os

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as t
import numpy as np


__all__ = ['MyDataset', 'plot_grid_images']


class MyDataset(Dataset):
    def __init__(self, data_dir, data_type, transform=None, target_transform=None):
        super().__init__()
        self.data_dir = data_dir ## path to the data directory
        self.data_type = data_type ## train data or test data

        self.image_names, self.labels = self.__process_data() ##get the image_names (files) and labels


        self.transform = transform ### transform to be applied on the images
        self.target_transform = target_transform ## transform to be applied on the labels

    def __len__(self):
        return len(self.image_names) ## return th length of the dataset
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx] ### get the image at the index idx
        ##new added code
        # print(f"Loading image: {image_name}")

        label = self.labels[idx] ### get label at the index idx

        label_map = {0: 'CAT', 1: 'DOG'} ###map the label to the folder name

        image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name) ## create the path to the image
        image = Image.open(image_path) ## read the image

        ###new added code
        #print(f"Image shape: {image.size}, Label: {label_map}")

        ##convert from PIL to tensor
        # image = torch.from_numpy(np.array(image))

        if self.transform: ##apply transform function if exists
            image = self.transform(image)


        if self.target_transform: ## apply target transform if exists
            label = self.target_transform(label)

        return image, torch.tensor(label).long() ## return the image and label
        

    def __process_data(self): ## This function will process the data and return the image_names and labels
        cat_images = os.listdir(os.path.join(self.data_dir, self.data_type, 'CAT'))  ## list of alll the images in the cat folder
        dog_images = os.listdir(os.path.join(self.data_dir, self.data_type, 'DOG')) ## list of all the images in the dog folder

        ## check if the images are in jpg only
        cat_images = [image_name for image_name in cat_images if '.jpg' in image_name] ##filter out the images which are not jpg
        dog_images = [image_name for image_name in dog_images if '.jpg' in image_name] ##filter out the images which are not jpg

        combined_images = cat_images + dog_images ## combine cat and dog images
        labels = [0]*len(cat_images) + [1]*len(dog_images) ## create the labels for the images

        return combined_images, labels
    

def plot_grid_images(images, labels, batch_size): ## this function will plot the grid of images
    fig, axes = plt.subplots(batch_size//4, 4, figsize=(20, 20))
    for idx, image in enumerate(images):
        image = image.permute(1, 2, 0)
        images, labels = images, labels
        row = idx//4
        col = idx%4 ###?
        axes[row, col].imshow(image)
        axes[row, col].set_title(labels[idx].item())
        axes[row, col].axis('off')
    plt.show()

# if __name__ == "__main__":
#     data_dir = 'Images'
#     data_type = 'train'

#     batch_size = 16

#     ## define the transforms to be applied on the images
#     transforms = t.Compose([t.Resize((224, 224)),
#                             t.ToTensor()])

#     ## create the dataset
#     dataset = MyDataset(data_dir, data_type, transform=transforms)

#     ##create the dataloader
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     for x, y in dataloader:
#         print(x.shape, y.shape)
#         plot_grid_images(x,y, batch_size)
#         break
