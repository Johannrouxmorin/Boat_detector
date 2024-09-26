# Semantic segmentation trained neural network inference pipeline for boat detection

This section explains how to use this neural network inference pipeline trained by semantic segmentation on never-before-seen Test images using the open source Pytorch Lightning library (see doc: https://lightning.ai/docs/pytorch/stable/). 

## 1. Loading input images

In order to load images into arrays of different dimensions (N, H, W, C), I've implemented a “load_images” function that takes all images directly into a Test folder.

```python
images_dir = functions_path + '/Datas/images_directory/'
images, pre_event, post_event = load_images(images_dir, 'png' or 'tif' or 'tiff')
```

## 2. Neural network characteristics

If you have trained a neural network with the “Train_boat_detector.ipynb” notebook, you can copy the “model”, “data_augmentation”, “log_dir”, “logger” and “segmentation_model” variables according to the training characteristics you have chosen (see Train_boat_detector.ipynb).

## 3. Loading weights and predictions

To load model weights, you need to fill in a file in .ckpt format and apply it to the segmentation model. Evaluation of the segmentation on images is performed using the “evaluation” function, which asks for image dimensions (N, H, W, C), the segmentation model and whether we want binary prediction (only 0 and 1 with a threshold) or sigmoid prediction (enter “False” in the function).

```python
checkpoint = torch.load('../Weights/my_best_model.ckpt')
segmentation_model.load_state_dict(checkpoint)
segmentation_model.eval()

images, predicted_masks = evaluation(images, model=segmentation_model, binary=True or False)
```