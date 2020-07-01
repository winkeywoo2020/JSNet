# JSNet：A simulation network of JPEG lossy compression for color images
## Overview
#### JSNet is a JPEG simulation network which is proposed to reappear the whole procedure of the JPEG lossy compression and restoration except entropy encoding as realistically as possible for color images. The steps of sampling, DCT, and quantization are modelled by the max-pooling layer, convolution layer, and 3D noise-mask, respectively. The proposed JSNet can simulate JPEG lossy compression with any quality factors. 
[JSNet: A simulation network of JPEG lossy compression and restoration for robust image watermarking against JPEG attack](https://www.sciencedirect.com/science/article/abs/pii/S1077314220300783)

## Prerequisites
#### Linux
#### NVIDIA GPU+CUDA CuDNN (CPU mode may also work, but untested)
#### Install Tensorflow and dependencies
#### Clone the repo: [JSNet](https://github.com/winkeywoo2020/JSNet)

## Setup training dataset
#### To train JSNet, you need to create two data folders TrainRaw and TrainGoal. The TrainRaw folder contains the original images without JPEG compression. The TrainGoal folder contains the JPEG compressed images of that in TrainRaw folder. 
#### You can randomly select 6000 color images from the [ImageNet](http://www.image-net.org/). The dataset used for training was collected by other researches, please cite their paper if you use the data for your own research.
#### All the images are cropped to 256*256. You can refer to the Data preprocessing folder in the repository to prepare your own dataset.

## Using Pre-trained Models
#### As the training images are randomly selected, for better and faster training, we release the pre-trained model. You can load the pre-trained model for fine-tuning in the training stage.

## Training and Test Details
#### To train a model, you need to prepare two TXT files for data loading and you are welcome to modify the source code using other data loading methods. Just run the JSNet90_train_ImageNet.py on your machine for training. The same goes for testing. In the training stage, you may adjust parameters for a better model. You can also apply the trained model in the CRWNet if you are interested in watermarking. We provide two versions, the basic CRWNet and the CRWNet90.
### Warm Tips: 
#### 1)	Please pay attention to checking whether the original images are correspond to the target images, otherwise the training will fail.
#### 2)	If you want to simulate JPEG with other quality factors, you need to make two changes:
* a) modify your TrainGoal folder with the target compressed images;
* b) modify the corresponding network parameters.

## Related Works
#### 1) [Hidden: Hiding data with deep networks](https://openaccess.thecvf.com/content_ECCV_2018/html/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.html) 
#### 2) [ReDMark: framework for residual diffusion watermarking based on deep networks](https://www.sciencedirect.com/science/article/abs/pii/S0957417419308759)







