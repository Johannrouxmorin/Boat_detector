import os 
import numpy as np 
from tqdm import tqdm 
from PIL import Image
import matplotlib.pyplot as plt

def pad_image(image):
    H, W, C = image.shape
    
    H_target = ((H + 31) // 32) * 32
    W_target = ((W + 31) // 32) * 32

    pad_h = H_target - H
    pad_w = W_target - W
    
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    
    return padded_image

def load_data_airbus(image_dir):
    msk_path = os.path.join(image_dir, 'masks_v2')
    img_path = os.path.join(image_dir, 'train_v2')
    images = []
    masks = []

    image_filenames = sorted(os.listdir(img_path))
    mask_filenames = sorted(os.listdir(msk_path))

    assert len(image_filenames) == len(mask_filenames), "Nombre d'images et de masques différent!"

    for img_name, msk_name in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames), desc="Loading Airbus images and masks") :
        if img_name.split('.')[0] == msk_name.split('.')[0]:
            # Charger l'image et le masque
            image_path = os.path.join(img_path, str(img_name))
            image = normalize_image(image_path)
            images.append(image)

            mask_path = os.path.join(msk_path, str(msk_name))
            mask = plt.imread(mask_path)
            mask = (mask > 0.5).astype(np.uint8)
            masks.append(mask)
        else:
            print(f"Image {img_name} et masque {msk_name} ne correspondent pas, ignoré.")

    image_list = []
    mask_list = []
    for i in range(len(images)) :
        if images[i].shape == (768, 768, 3) :
            top_left_i, top_left_m = images[i][:512, :512, :3], masks[i][:512, :512]

            all_zeros_tl = np.all(top_left_m == 0)

            if all_zeros_tl == False : 
                image_list.append(top_left_i)
                mask_list.append(top_left_m)
        else :
            continue

    images = np.stack(image_list, axis=0)
    masks = np.stack(mask_list, axis=0)

    masks = np.expand_dims(masks, axis=-1)

    return np.array(images), np.array(masks)

def normalize_image(image_path):
    """
    Normalise une image RGB en une matrice avec des valeurs entre 0 et 1.
    
    Paramètres :
    - image_path (str) : chemin de l'image .jpg
    
    Retourne :
    - img_normalized (numpy array) : image normalisée avec des valeurs entre 0 et 1
    """
    # Ouvrir l'image en mode RGB
    img = Image.open(image_path).convert('RGB')
    
    # Convertir l'image en un tableau NumPy
    img_array = np.asarray(img, dtype=np.float32)
    
    # Normaliser l'image en divisant par 255 (valeur maximale pour une image 8-bit)
    img_normalized = img_array / 255.0
    
    return img_normalized

def merge2datasets(images1, masks1, images2, masks2) :
    images, masks = [], []
    for i in range(len(images1)) :
        images.append(images1[i])
        masks.append(masks1[i])

    for j in range(len(images2)) :
        images.append(images2[j])
        masks.append(masks2[j])

    return np.array(images), np.array(masks)

def plot_segmentation_results(input_images, predicted_masks, num_images=1):
    plt.subplot(num_images, 2, 1)
    plt.imshow(input_images.transpose(1, 2, 0))
    plt.title("Image d'entrée")
    plt.axis('off')

    plt.subplot(num_images, 2, 2)
    plt.imshow(predicted_masks, cmap='gray')
    plt.title("Masque prédit")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def load_images(image_dir, images_type) :
    images = []
    pre_event_images, post_event_images = [], []
    image_filenames = os.listdir(image_dir)
    for img_name in tqdm(image_filenames, total=len(image_filenames), desc="Loading images") :
        image_path = os.path.join(image_dir, str(img_name))
        if images_type == 'tif' or images_type == 'tiff' :
            with rasterio.open(image_path) as img_file:
                image = img_file.read()  # Lecture des données de l'image TIFF
                image = reshape_as_image(image)  # Conversion en tableau numpy 2D ou 3D
                images.append(image[:,:,:3])
        elif images_type == 'png' :
            image = plt.imread(image_path)
            images.append(image[:,:,:3])
        elif images_type == 'jpg' :
            image = normalize_image(image_path)
            images.append(image[:,:,:3])

    return np.array(images)