import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np

dataset_dir = "/scratch/t.tovi/datasets/"
image_dir = "COCO_stuff_images/train2017/"
annotation_dir = "COCO_stuff_annotations/train2017/"

class COCOStuffDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
            annotation_dir (string): Directory with all the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transforms.ToTensor()
        self.images = os.listdir(image_dir)
        self.default_size = 224
    
    def random_crop(self, image, mask, output_size):
        """
        Applies the same random crop to both image and mask.
        
        :param image: PIL.Image, the input image.
        :param mask: PIL.Image, the corresponding segmentation mask.
        :param output_size: tuple or int, the desired output size.
        :return: PIL.Image, PIL.Image, the cropped image and mask.
        """
        transform = transforms.CenterCrop(224)
        image_cropped = transform(image)
        mask_cropped = transform(mask)
        return image_cropped, mask_cropped

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        # Load annotation
        annotation_name = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '.png'))
        annotation = Image.open(annotation_name)

        # Load the label mapping
        digit_to_object_mapping = {}
        with open('src/labels.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                digit_to_object_mapping[int(key)] = value.strip()
        digit_to_object_mapping[255] = "unlabled"

        # Crop both images
        image, mask = self.random_crop(image, annotation, self.default_size)
        #mask = annotation

        image = np.array(image)
        mask = np.array(mask)

        # Id
        ids = np.unique(mask)
        labels = [digit_to_object_mapping[id] for id in ids]

        # Binary masks
        masks = np.stack([mask==id for id in ids])

        sample = {'image': image, 'annotation': masks, 'labels': labels}

        return sample