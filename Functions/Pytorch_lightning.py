import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import random
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid

from CNN_models import CNN_9, CNN_36, CNN_145, CNN_581
from data_augmentation import DataAugmentation
from functions import pad_image, plot_segmentation_results
from metrics import iou_score, DiceLoss, CombinedLoss, JaccardLoss, ContrastiveLoss, FocalLoss

class CustomDataset(Dataset):
    def __init__(self, images, masks, data_augmentation):
        self.images = images
        self.masks = masks
        self.target_size = data_augmentation['RandomCrop']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        threshold = 0
        mask = (mask > threshold).astype(np.float32)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        image_crop, mask_crop = self.random_crop(image_tensor, mask_tensor)

        return image_crop, mask_crop

    def random_crop(self, image, mask):
        h, w = self.target_size
        th, tw = image.size(1), image.size(2)
        if w == tw and h == th:
            return image, mask
        i = torch.randint(0, th - h + 1, (1,))
        j = torch.randint(0, tw - w + 1, (1,))
        image_crop = TF.crop(image, i.item(), j.item(), h, w)
        mask_crop = TF.crop(mask, i.item(), j.item(), h, w)
        return image_crop, mask_crop

class DataModule(pl.LightningDataModule):
    def __init__(self, images, masks, data_augmentation, batch_size=32):
        super().__init__()
        self.images = images
        self.masks = masks
        self.data_augmentation = data_augmentation
        self.batch_size = batch_size

    def setup(self, stage=None):
        total_size = len(self.images)
        train_size = int(0.85 * total_size)
        val_size = total_size - train_size

        self.train_dataset = CustomDataset(self.images[:train_size], self.masks[:train_size], self.data_augmentation)
        self.val_dataset = CustomDataset(self.images[train_size:], self.masks[train_size:], self.data_augmentation)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=10)

class SegmentationModel(pl.LightningModule):
    def __init__(self, model=None, optimizer_name=None, learning_rate=None, log_every_n_steps=None, loss=None, data_augmentation=None, log_dir=None, logger=None, **optimizer_params):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.optimizer_params = optimizer_params
        self.log_every_n_steps = log_every_n_steps
        self.automatic_optimization = True
        self.data_augmentation = data_augmentation

        self.training_step_outputs_loss = []
        self.training_step_outputs_iou = []

        self.transform = DataAugmentation(data_augmentation)
        self.model = model

        if loss == 'DiceLoss' :
            self.loss = DiceLoss()
        elif loss == 'bce_loss' :
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == 'CombinedLoss' :
            self.loss = CombinedLoss()
        elif loss == 'JaccardLoss' :
            self.loss = JaccardLoss()
        elif loss == 'ContrastiveLoss' :
            self.loss = ContrastiveLoss()
        elif loss == 'FocalLoss' :
            self.loss = FocalLoss(logits=True)

        self.log_dir = log_dir
        self.tb_writer = logger

    def configure_optimizers(self):
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        elif self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

        # Change the learning rate
        scheduler = CosineAnnealingLR(optimizer, T_max=20)

        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training:
            x, y = self.transform(x, y)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        iou = iou_score(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)

        self.training_step_outputs_loss.append(loss)
        self.training_step_outputs_iou.append(iou)

        self.log_images(x, y, y_hat, 'train', self.global_step)

        return {"loss": loss, "iou": iou}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        val_loss = self.loss(y_hat, y)
        iou = iou_score(y_hat, y)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

        self.log_images(x, y, y_hat, 'val', self.global_step)

        return {"val_loss": val_loss, "val_iou": iou}

    def log_images(self, x, y, y_hat, tag, step):
        # Création grille d'image
        input_grid = make_grid(x[:3], nrow=3, normalize=True)
        target_grid = make_grid(y[:3], nrow=3, normalize=True)
        predicted_grid = make_grid(y_hat[:3], nrow=3, normalize=True)

        # Log images using SummaryWriter's
        self.tb_writer.experiment.add_image(f'{tag}/input_images', input_grid, global_step=step)
        self.tb_writer.experiment.add_image(f'{tag}/target_masks', target_grid, global_step=step)
        self.tb_writer.experiment.add_image(f'{tag}/predicted_masks', predicted_grid, global_step=step)

    def on_train_epoch_end(self):
        epoch_mean_loss = torch.stack(self.training_step_outputs_loss).mean()
        self.log("loss", epoch_mean_loss, prog_bar=True)

        epoch_mean_iou = torch.stack(self.training_step_outputs_iou).mean()
        self.log("iou", epoch_mean_iou, prog_bar=True)

        self.training_step_outputs_loss.clear()
        self.training_step_outputs_iou.clear()

def evaluate_model_performance(model, data_module, num_samples=5, threshold=0.5):
    val_dataloader = data_module.val_dataloader()
    val_indices = list(range(len(val_dataloader.dataset)))
    random_val_indices = random.sample(val_indices, num_samples)

    f1_scores = []
    recall_scores = []
    iou_scores = []

    model.eval()
    with torch.no_grad():
        for idx in random_val_indices:
            
            val_images, val_masks = val_dataloader.dataset[idx]

            # Prédire les masques
            predicted_masks = model(val_images.unsqueeze(0))  # Ajouter une dimension pour le batch
            predicted_mask_binary = (predicted_masks > threshold).float()

            # Aplatir les masques pour le calcul des scores
            predicted_mask_flat = predicted_mask_binary.cpu().numpy().flatten()
            true_mask_flat = val_masks.cpu().numpy().flatten()

            # Calcul du F1-score et du Recall
            f1 = f1_score(true_mask_flat, predicted_mask_flat, zero_division=1)
            recall = recall_score(true_mask_flat, predicted_mask_flat, zero_division=1)

            # Calcul de l'IoU
            intersection = (predicted_mask_flat * true_mask_flat).sum()
            union = predicted_mask_flat.sum() + true_mask_flat.sum() - intersection
            iou = intersection / union if union != 0 else 0

            # Ajouter les scores aux listes
            f1_scores.append(f1)
            recall_scores.append(recall)
            iou_scores.append(iou)

            # Afficher l'image, le masque prédit et le masque réel
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(val_images.permute(1, 2, 0)) 
            plt.title("Image")

            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask_binary.squeeze().cpu().numpy(), cmap='gray')
            plt.title("Masque prédit")

            plt.subplot(1, 3, 2)
            plt.imshow(val_masks.squeeze(), cmap='gray') 
            plt.title("Masque réel")

            plt.show()

    # Calcul des moyennes
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

    # Afficher les résultats
    print(f"Mean F1-score : {mean_f1}")
    print(f"Mean Recall : {mean_recall}")
    print(f"Mean IoU : {mean_iou}")

    return {
        'mean_f1': mean_f1,
        'mean_recall': mean_recall,
        'mean_iou': mean_iou
    }

def evaluation(images, model=None, binary=True) :
    input_images = np.array([pad_image(img) for img in images])
    input_images = input_images.transpose(0, 3, 1, 2)
    input_images = torch.from_numpy(input_images).float()
    images_predicted_masks = []
    for i in tqdm(range(len(input_images)), desc="Loading predictions") :
        with torch.no_grad() :
            img = input_images[i].unsqueeze(0)
            predicted_masks = model(img)
            predicted_masks = predicted_masks.squeeze().cpu().numpy()

        if binary == True :
            predicted_masks = (predicted_masks > 0.5).astype(np.uint8)
            images_predicted_masks.append(predicted_masks)
            plot_segmentation_results(input_images[i].cpu().numpy(), predicted_masks)
        else :
            images_predicted_masks.append(predicted_masks)
            plot_segmentation_results(input_images[i].cpu().numpy(), predicted_masks)
    
    return input_images, images_predicted_masks

def get_model(num_classes, neural_network, encoder_name, encoder_depth, activation):
    if neural_network == 'Unet' :
        model = smp.Unet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'UnetPlusPlus' :
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'DeepLabV3' :
        model = smp.DeepLabV3(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'DeepLabV3Plus' :
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'MAnet' :
        model = smp.MAnet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'Linknet' :
        model = smp.Linknet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'FPN' :
        model = smp.FPN(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'PSPNet' :
        model = smp.PSPNet(encoder_name=encoder_name, encoder_depth=encoder_depth, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'PAN' :
        model = smp.PAN(encoder_name=encoder_name, in_channels=3, classes=1, activation=activation)
    elif neural_network == 'CNN_9':
        model = CNN_9(num_classes=num_classes)
    elif neural_network == 'CNN_36':
        model = CNN_36(num_classes=num_classes)
    elif neural_network == 'CNN_145':
        model = CNN_145(num_classes=num_classes)
    elif neural_network == 'CNN_581':
        model = CNN_581(num_classes=num_classes)

    return model