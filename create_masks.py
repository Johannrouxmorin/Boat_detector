import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Fonction pour décoder RLE en masques binaires
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    try:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T 
    except ValueError:
        return None

# Fonction principale
def create_masks(train_dir, masks_dir, csv_file):
    os.makedirs(masks_dir, exist_ok=True)

    train = os.listdir(train_dir)
    masks = pd.read_csv(csv_file, sep=';')

    corrupted_images = []

    # Parcours de toutes les images dans le dossier train_v2
    for ImageId in tqdm(train, total=len(train), desc="Creating masks"):
        img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

        all_masks = np.zeros((768, 768), dtype=np.uint8)

        corrupted = False
        for mask in img_masks:
            if pd.isnull(mask): 
                continue
            decoded_mask = rle_decode(mask)
            if decoded_mask is None:
                corrupted = True 
                break 
            all_masks += decoded_mask

        if corrupted:
            print(f"Corrupted mask for image {ImageId}")
            corrupted_images.append(ImageId)
            continue 

    
        all_masks = np.clip(all_masks, 0, 1)

        mask_image = Image.fromarray((all_masks * 255).astype(np.uint8)) 
        mask_image.save(os.path.join(masks_dir, ImageId.replace('.jpg', '.png')))

    print('All masks have been created and saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create segmentation masks from images and CSV files.')
    parser.add_argument('train_dir', type=str, help='Directory containing images to be segmented.')
    parser.add_argument('masks_dir', type=str, help='Directory where to save generated masks.')
    parser.add_argument('csv_file', type=str, help='Path to CSV file containing masks.')

    args = parser.parse_args()
    
    create_masks(args.train_dir, args.masks_dir, args.csv_file)