import torch
import torch.nn as nn
from torch import Tensor
import kornia.augmentation as K 
from typing import Tuple

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, data_augmentation) :
        super(DataAugmentation, self).__init__()

        # Initialize geometric transformations list
        geometric_transformations = []
        if 'RandomRotation' in data_augmentation:
            geometric_transformations.append(K.RandomRotation(degrees=data_augmentation['RandomRotation']))
        if 'RandomVerticalFlip' in data_augmentation:
            geometric_transformations.append(K.RandomVerticalFlip(p=data_augmentation['RandomVerticalFlip']))
        if 'RandomHorizontalFlip' in data_augmentation:
            geometric_transformations.append(K.RandomHorizontalFlip(p=data_augmentation['RandomHorizontalFlip']))
        if 'RandomCrop' in data_augmentation:
            self.crop_size = data_augmentation['RandomCrop']
            geometric_transformations.append(K.RandomCrop(size=self.crop_size))
        if 'RandomAffine' in data_augmentation:
            geometric_transformations.append(K.RandomAffine(
                degrees=data_augmentation['RandomAffine']['degrees'],
                translate=data_augmentation['RandomAffine']['translate'],
                scale=data_augmentation['RandomAffine']['scale'],
                shear=data_augmentation['RandomAffine']['shear'],
                p=data_augmentation['RandomAffine']['p']
            ))
        if 'RandomPerspective' in data_augmentation:
            geometric_transformations.append(K.RandomPerspective(
                distortion_scale=data_augmentation['RandomPerspective']['distortion_scale'],
                p=data_augmentation['RandomPerspective']['p']
            ))

        self.geometric_transform = nn.Sequential(*geometric_transformations)

        # Initialize radiometric transformations list
        radiometric_transformations = []
        if 'ColorJitter' in data_augmentation:
            radiometric_transformations.append(K.ColorJitter(
                brightness=data_augmentation['ColorJitter']['brightness'],
                contrast=data_augmentation['ColorJitter']['contrast'],
                saturation=data_augmentation['ColorJitter']['saturation'],
                hue=data_augmentation['ColorJitter']['hue'],
                p=data_augmentation['ColorJitter']['p']
            ))
        if 'RandomGamma' in data_augmentation:
            radiometric_transformations.append(K.RandomGamma(
                gamma=data_augmentation['RandomGamma']['gamma'],
                p=data_augmentation['RandomGamma']['p']
            ))
        if 'RandomGrayscale' in data_augmentation:
            radiometric_transformations.append(K.RandomGrayscale(p=data_augmentation['RandomGrayscale']))
        if 'RandomBrightness' in data_augmentation:
            radiometric_transformations.append(K.RandomBrightness(
                brightness=data_augmentation['RandomBrightness']['brightness'],
                p=data_augmentation['RandomBrightness']['p']
            ))
        if 'RandomContrast' in data_augmentation:
            radiometric_transformations.append(K.RandomContrast(
                contrast=data_augmentation['RandomContrast']['contrast'],
                p=data_augmentation['RandomContrast']['p']
            ))
        if 'RandomSharpness' in data_augmentation:
            radiometric_transformations.append(K.RandomSharpness(
                sharpness=data_augmentation['RandomSharpness']['sharpness'],
                p=data_augmentation['RandomSharpness']['p']
            ))
        if 'GaussianBlur' in data_augmentation:
            radiometric_transformations.append(K.RandomGaussianBlur(
                kernel_size=data_augmentation['GaussianBlur']['kernel_size'],
                sigma=data_augmentation['GaussianBlur']['sigma'],
                p=data_augmentation['GaussianBlur']['p']
            ))

        self.radiometric_transform = nn.Sequential(*radiometric_transformations)

    def forward(self, images: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        seed = torch.randint(0, 1000000, (1,))
        torch.manual_seed(seed)

        # Concatenate images and masks along the channel dimension
        inputs = torch.cat((images, masks), dim=1)

        # Apply the same random transformation to both images and masks
        transformed_inputs = self.transform(inputs, seed.item())

        # Split the transformed inputs back into images and masks
        transformed_images, transformed_masks = torch.split(transformed_inputs, images.shape[1], dim=1)

        # Apply radiometric transformations only on images
        transformed_images = self.radiometric_transform(transformed_images)

        return transformed_images, transformed_masks

    def transform(self, inputs: Tensor, seed: int) -> Tensor:
        # Apply the random geometric transformations
        random_transformed = self.geometric_transform(inputs)
        return random_transformed