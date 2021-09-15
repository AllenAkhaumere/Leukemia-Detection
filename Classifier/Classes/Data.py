#############################################################################################
#
# Project:       Mascot Defense Research Project
# Repository:    ALL Detection System 2020
# Project:       AllDS2020 Classifier
#
# Author:        Allen Akhaumere
# Contributors:
# Title:         Data Class
# Description:   Data class for the Acute Lymphoblastic Leukemia Pytorch CNN Classifier
#                ALL Classifier.
# License:       MIT License
# Last Modified: 2020-07-23
#
############################################################################################

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets




class LeukemiaDataset(Dataset):

    """
    Acute Lymphoblastic Leukemia Dataset Reader.

     Args:
           df_data: Dataframe for CSV file
           data_dir: path to Lymphoblastic Leukemia Data
           transform: transforms for performing data augmentation
    """

    def __init__(self, df_data, data_dir='./', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name + '.jpg')
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def augmentation():
    """Acute Lymphoblastic Leukemia data augmentation"""
    mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    training_transforms = transforms.Compose([transforms.Resize((100, 100)),
                                              transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(100),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomGrayscale(p=0.1),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std_dev)])

    validation_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)])

    return training_transforms, validation_transforms


# Load the datasets with ImageFolder
def load_datasets(train_dir, training_transforms, valid_dir, validation_transforms):
    """ """
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    return training_dataset, validation_dataset