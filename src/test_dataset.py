import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np

dataset_dir = "../datasets/VOC2012/"
image_dir = "JPEGImages/"
annotation_dir = "Annotations/"


class PascalVOC(Dataset):
    def __init__(self, image_dir, annotation_dir, img_size=224):
        """
        Args:
            image_dir (string): Directory with all the images.
            annotation_dir (string): Directory with all the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = os.listdir(image_dir)
        self.img_size = 224

    def center_crop(self, image, mask):
        transform = transforms.CenterCrop(self.img_size)
        return transform(image), transform(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        # Load annotation
        annotation_name = os.path.join(
            self.annotation_dir, self.images[idx].replace(".jpg", ".png")
        )
        annotation = Image.open(annotation_name)

        # Load the label mapping
        digit_to_object_mapping = {}
        with open("src/labels.txt", "r") as file:
            for line in file:
                key, value = line.strip().split(":")
                digit_to_object_mapping[int(key)] = value.strip()
        digit_to_object_mapping[255] = "unlabled"

        image, mask = self.center_crop(image, annotation)
        mask = np.array(mask)

        # Get ids and labels
        ids = np.unique(mask)
        labels = [digit_to_object_mapping[id] for id in ids]

        # Binary masks
        nonempty_masks = [mask == id for id in ids]

        sample = {"image": image, "annotation": nonempty_masks, "labels": labels}

        return sample
