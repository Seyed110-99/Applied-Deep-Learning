# # oxford_pet_dataset.py

# import os
# import xml.etree.ElementTree as ET
# from pathlib import Path
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# from torchvision.datasets import OxfordIIITPet

# def load_bounding_boxes(xml_dir):
#     bbox_dict = {}
#     for fname in os.listdir(xml_dir):
#         if not fname.endswith('.xml'): continue
#         path = os.path.join(xml_dir, fname)
#         tree = ET.parse(path)
#         root = tree.getroot()

#         image_id = root.find('filename').text.split('.')[0]
#         obj = root.find('object')
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         ymin = int(bbox.find('ymin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymax = int(bbox.find('ymax').text)

#         bbox_dict[image_id] = (xmin, ymin, xmax, ymax)
#     return bbox_dict

# class OxfordPetDataset(Dataset):
#     def __init__(self, root_dir='data', split='test', crop_bbox=False, return_mask=False, image_size=(224, 224)):
#         self.root_dir = root_dir
#         self.split = split
#         self.crop_bbox = crop_bbox
#         self.return_mask = return_mask
#         self.image_size = image_size

#         self.img_transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.ToTensor()
#         ])

#         self.mask_transform = transforms.Compose([
#             transforms.Resize(image_size, interpolation=Image.NEAREST),
#             transforms.ToTensor()
#         ])

#         torch_split = 'trainval' if split == 'train' else 'test'
#         dataset = OxfordIIITPet(
#             root=root_dir,
#             split=torch_split,
#             target_types='category',
#             transform=None,
#             download=True
#         )

#         self.image_paths = dataset._images
#         self.labels = dataset._labels

#         if crop_bbox:
#             self.bbox_dict = load_bounding_boxes(os.path.join(root_dir, 'annotations/xmls'))

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image_id = Path(image_path).stem
#         image = Image.open(image_path).convert('RGB')

#         if self.crop_bbox and hasattr(self, 'bbox_dict') and image_id in self.bbox_dict:
#             xmin, ymin, xmax, ymax = self.bbox_dict[image_id]
#             image = image.crop((xmin, ymin, xmax, ymax))

#         image = self.img_transform(image)
#         label = self.labels[idx]

#         if self.return_mask:
#             mask_path = (str(image_path)
#                          .replace("images", "annotations/trimaps")
#                          .replace(".jpg", ".png"))
#             mask = Image.open(mask_path)
#             mask = self.mask_transform(mask)
#             mask = (mask * 255).long().squeeze(0)
#             return image, label, mask
#         else:
#             return image, label

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.datasets import OxfordIIITPet

class OxfordPetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',           
        return_mask: bool = False,
        image_size=(224,224),
        img_transform=None,
        mask_transform=None
    ):
        self.root_dir     = str(root_dir)
        self.split        = split
        self.return_mask  = return_mask

        # assign transforms (or fall back to simple resize+to-tensor)
        if img_transform is None:
            from torchvision import transforms
            self.img_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.img_transform = img_transform

        if mask_transform is None:
            from torchvision import transforms
            self.mask_transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = mask_transform

        # load the standard Oxford-IIIT Pet splits
        torch_split = 'trainval' if split=='train' else 'test'
        base = OxfordIIITPet(
            root=self.root_dir,
            split=torch_split,
            target_types='category',
            download=True,
            transform=None
        )
        self.image_paths = base._images
        self.labels      = base._labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert('RGB')
        img_t = self.img_transform(img)

        if not self.return_mask:
            return img_t, label

        # if return_mask=True, load the corresponding trimap and map to {0,1,2}
        mask_path = path.replace("images", "annotations/trimaps").replace(".jpg", ".png")
        mask = Image.open(mask_path)
        mask_t = self.mask_transform(mask).long().squeeze(0)
        return img_t, label, mask_t


def construct_dataset(
    data_dir,
    train_frac: float,
    train_transforms,
    test_transforms,
    image_size=(256,256),
    crop_bbox=False        # accepted but ignored
):
    """
    Returns (trainset, testset):
     - trainset is a random split of the 'trainval' subset,
     - testset is the full 'test' subset.
    """
    # build full trainval dataset (with masks)
    full = OxfordPetDataset(
        root_dir=str(data_dir),
        split='train',
        return_mask=True,
        image_size=image_size,
        img_transform=train_transforms,
        mask_transform=train_transforms
    )
    n = len(full)
    n_train = int(train_frac * n)
    n_val   = n - n_train

    train_ds, _ = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # full test set
    test_ds = OxfordPetDataset(
        root_dir=str(data_dir),
        split='test',
        return_mask=True,
        image_size=image_size,
        img_transform=test_transforms,
        mask_transform=test_transforms
    )

    return train_ds, test_ds