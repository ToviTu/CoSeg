import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np

src_dir = "/home/research/jianhong.t/OpenVocab_Seg_with_AutoRegres/src/"
dataset_dir = "/scratch/t.tovi/datasets/"
image_dir = "COCO_stuff_images/train2017/"
annotation_dir = "COCO_stuff_annotations/train2017/"

class COCOStuffDataset(Dataset):
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
    
    def resize(self, image, mask):
        transform = transforms.CenterCrop(min(image.size[1:]))

        cropped_image = transform(image)
        cropped_mask = transform(mask)

        resized_image = transforms.Resize((self.img_size, self.img_size),transforms.InterpolationMode.BILINEAR)(image)
        resized_mask= transforms.Resize((self.img_size, self.img_size), transforms.InterpolationMode.NEAREST)(mask)
        return resized_image, resized_mask

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
        with open(f'{src_dir}labels.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                digit_to_object_mapping[int(key)] = value.strip()
        digit_to_object_mapping[255] = "unlabled"

        image, mask = self.resize(image, annotation)
        #image, mask = self.center_crop(image, annotation)
        mask = np.array(mask)

        # Get ids and labels
        ids = np.unique(mask)
        labels = [digit_to_object_mapping[id] for id in ids]

        # Binary masks
        nonempty_masks = [mask==id for id in ids]

        sample = {'image': image, 'annotation': nonempty_masks, 'labels': labels}

        return sample

def collate_fn_factory(processor, clip_text_encoder):

    def collate_fn(batch):
        size = processor.image_processor.size['shortest_edge'] #224
        transform = transforms.ToTensor()

        # Preprocess pixel values
        images = [each['image'] for each in batch]
        batch_pixel_values = processor(None, images=images, return_tensors='pt')['pixel_values']

        # Preprocess texts
        batch_input_ids = processor(
            [" ".join(entry['labels']) for entry in batch],
            padding=True,
            return_tensors='pt'
        )['input_ids']

        # Preprocess masks
        max_size = batch_input_ids.shape[1]
        batch_masks = np.stack([
            np.stack([np.zeros((size, size))] + each['annotation'] + [np.zeros((size, size))] * (max_size - len(each['annotation']) - 1) )
            for each in batch
        ])
        batch_masks = torch.tensor(batch_masks)

        return {
            "pixel_values": batch_pixel_values, 
            "masks": batch_masks.type(torch.float32),
            "input_ids": batch_input_ids,
            "token_summary_idx": torch.tensor(batch_token_summary)
        }
    
    return collate_fn



        

