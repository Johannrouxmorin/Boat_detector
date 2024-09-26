# Semantic segmentation neural network architecture training pipeline for boat detection

This section explains how to use this semantic segmentation neural network architecture training pipeline using the open source Pytorch Lightning library (see doc: https://lightning.ai/docs/pytorch/stable/). The advantage of this training pipeline is that it can be adapted for different input images and masks, and different neural network architecture characteristics can be chosen.

## 1. Input data

Before we can start training, it's important to import, process and distribute our dataset according to our final objective. To do this, we need to create a function that returns images and masks, arrays of dimensions (N, H, W, C). 

### Airbus Ship Detection Challenge

The Airbus boat detection challenge data are satellite images applied to the observation of boats from space. They are composed of RGB images (1024,1024) and annotation polygons of the boats visible on the images. You can download the data via this link (https://www.kaggle.com/c/airbus-ship-detection/data), which includes almost 40,000 satellite images (around 30 GB of data). Once downloaded, it's important to create binary masks (1 if the pixels are part of the boat and 0 otherwise). To do this, we use the create_masks.py program, executing the following command in the terminal :

```bash
python create_masks.py Data_set/Airbus/train_v2/ Data_set/Airbus/masks_v2/ Data_set/Airbus/train_ship_segmentations_v2.csv
```

This command will create binary masks in a “masks_v2” directory from images stored in the “train_v2” directory, thanks to the “train_ship_segmentations_v2.csv” file showing where boats are located. Once the masks have been created, you can move on to the drive parameters stage. 

It is also possible to assemble data sets for more general drives. The following function creates two arrays consisting of images and masks from Dataset 1 and Dataset 2. A percentage is given at the output of this function to indicate the proportion of Dataset 1 and Dataset 2 in Dataset 1 + 2 (this is useful if you want to achieve transferability by running only Dataset 1 in training and Dataset 2 in validation).

```python
images, masks = merge2datasets(images_data1, masks_data1, images_data2, masks_data2)
```

## 2. Training parameters for neural networks 

Once the desired dataset has been pre-processed and selected, we can choose the neural network parameters and configure the model with Pytorch-lightning (https://lightning.ai/docs/pytorch/stable/).


### Model selection

The choice of model is important and has a major influence on segmentation results. The open source Segmentation_model_Pytorch library (see doc: https://smp.readthedocs.io/en/latest/index.html) provides a number of configurable neural network constructs.

```python
model = get_model(
    num_classes = n,        # Desired number of output classes from 1 to n
    # Select from
    neural_network = 'Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'PAN', 'CNN_9', 'CNN_36', 'CNN_145', 'CNN_581',
    # Select and download the encoder via this link: https://smp.readthedocs.io/en/latest/encoders.html
    encoder_name = 'ResNet', 'GERNet', 'DenseNet', 'EfficientNet', 'MobileNet', ...,
    encoder_depth = depth,      # Desired number of output classes from 1 to n
    # Select from
    activation = 'sigmoid', 'softmax', 'logsoftmax', 'tanh', 'identity'
)
```

### Data Augmentation

In order to generalize boat recognition and increase the performance of our model, it is possible to perform data augmentation by adding geometric or radiometric transformations to the training images and masks. It is also possible to make no transformations at all, by setting the probabilities to 0.

```python
data_augmentation = {
    # Geometric transformations
    'RandomRotation': (degrés_moins, degrés_plus),
    'RandomVerticalFlip': p,
    'RandomHorizontalFlip': p,
    'RandomCrop': (H, W),
    'RandomAffine': {
        'degrees': (degrés_moins, degrés_plus),  
        'translate': (pourcentage_moins, pourcentage_plus),
        'scale': (pourcentage_moins, pourcentage_plus),  
        'shear': (degrés_moins, degrés_plus),      
        'p' : p         
    },
    'RandomPerspective': {
        'distortion_scale': pourcentage,
        'p': p
    },

    # Radiometric transformations
    'ColorJitter': {
        'brightness': (pourcentage_moins, pourcentage_plus),
        'contrast': (pourcentage_moins, pourcentage_plus),
        'saturation': (pourcentage_moins, pourcentage_plus),
        'hue': (pourcentage_moins, pourcentage_plus),
        'p': p
    },
    'RandomGamma': {
        'gamma': (pourcentage_moins, pourcentage_plus),
        'p': p
    },
    'RandomGrayscale': p,
    'RandomBrightness': {
        'brightness': (pourcentage_moins, pourcentage_plus),
        'p': p
    },
    'RandomContrast': {
        'contrast': (pourcentage_moins, pourcentage_plus),
        'p': p
    },
    'RandomSharpness': {
        'sharpness': (pourcentage_moins, pourcentage_plus),
        'p': p
    },
    'GaussianBlur': {
        'kernel_size': (kernel_size, kernel_size),
        'sigma': (ecart_type_min, ecart_type_max),
        'p' : p
    }
}
```

### Segmentation model features

With Pytorch-lightning, the DataLoader module enables data to be downloaded and processed by applying transformations during training, and validation data to be distributed. The Tensorboard library lets you visualize the evolution of images over time, as well as evaluation parameters (F1-score, Recall, IoU, etc.) during training and validation phases.

```python
# Data module creation
data_module = DataModule(images, masks, data_augmentation, train_val_split=pourcentage)

log_dir = "../tensorboard/logs"
logger = pl.loggers.TensorBoardLogger(log_dir, name="my_training_directory")

# Segmentation model creation
segmentation_model = SegmentationModel(
    model = model,
    # Select from
    optimizer_name = 'Adam', 'AdamW', 'RMSprop', 'SGD',
    log_every_n_steps = n,
    learning_rate  = my_learning_rate,      # Typical values from 1e-1 to 1e-5 depending on performance
    # Select from
    loss = 'DiceLoss', 'bce_loss', 'CombinedLoss', 'JaccardLoss', 'ContrastiveLoss', 'FocalLoss',
    data_augmentation = data_augmentation,
    log_dir = log_dir,
    logger = logger
)
```

### Recording and using weights

In order to be able to use the network on test images, for example, and thus make inferences, it is important to be able to save the weights automatically. During training, Pytorch-lightning evaluates performance after each epoch and saves the best weights in a 'best_model.ckpt' file, depending on whether one of the training or validation metrics is better (smaller value = 'min' or larger value = 'max'). It is also possible to use pre-trained weights to reduce training time and improve performance using models we have already trained, for example.

```python
# Create a callback to save the best weights
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = 'my_metric',
    filename = 'best_model',
    save_top_k = number_of_best_model,
    mode = 'min' or 'max'
)

# Using pre-trained weights in a .ckpt file
name_best_model = 'Weights/my_model_wheights.ckpt'
segmentation_model.load_state_dict(torch.load(name_best_model))
```

### Training configuration

All that's left to do is configure the drive using the Pytorch-lightning library, which will run the dataset on a peripheral (CPU, GPU, TPU, etc.) up to a certain epoch limit. 

```python
# Training configuration
trainer = pl.Trainer(
    callbacks = [checkpoint_callback],
    max_epochs = max_epoch,
    log_every_n_steps = n_steps,
    logger = logger,
    devices = number_of_devices,
    precision = 16 or "16-mixed" or 32 or 64,
    accelerator = "auto" or "CPU" or "GPU"
)

# Clear GPU cache to free up more memory space
torch.cuda.empty_cache() 

# Model training with images and masks
trainer.fit(segmentation_model, datamodule=data_module)

# Save weights in a .ckpt file
torch.save(segmentation_model.state_dict(), 'best_model.ckpt')
```

### Performance visualization

In order to understand and observe the evolution of the training and validation process, we use Tensorboard, a useful open source tool for Tensorflow and Pytorch. This tool tracks metrics (losses, IoU, Recall, F1-score, ...), images transformed after data augmentation and the distribution of weights and gradients over time. To launch this tool, you need to go to the terminal to enable a :

```bash
tensorboard --logdir /tensorboard/logs/my_training_directory
```

## 3. Inference

Inference is used to evaluate the performance of the segmentation model on a random validation or test dataset. The following function displays a number of images, truth masks and associated prediction masks for visual inspection of model performance. It also calculates F1-score, Recall and IoU metrics by comparing predicted and actual masks.

```python
results = evaluate_model_performance(model, data_module, num_samples=5, threshold=0.5)
```

If you wish to perform inference on a dataset not used during training and validation, please take a look at the notebook structure in “Inference/inference_boat_detector.py”.