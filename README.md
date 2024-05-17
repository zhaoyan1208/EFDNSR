论文复审修改说明请点击[论文修改说明](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) 
# EFDNSR-PyTorch

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx 

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/zhaoyan1208/EFDNSR
cd EFDNSR
```
## Dataset
Due to the large size of the dataset, it cannot be directly uploaded to the webpage. Therefore, I have provided the code to download each dataset in the corresponding dataset folder.

## How to train

We used DIV2K and FFHQ dataset to train our model. 

The model files are uploaded! You can use the EDSR framework to train our RFDN.

# Facial recognition-tf2

## Dependencies

tensorflow==2.2.0

## Code

The trained weights can be downloaded from Baidu Cloud.
https://pan.baidu.com/s/1MZLAjBFXt1Oq1adxlztadw password: 3kwx

The CASIA WebFaces dataset for training and the LFW dataset for evaluation can be downloaded from Baidu Cloud.
https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw password: bcrq

## Assessment

Download the evaluation dataset, extract the LFW dataset used for evaluation, and place it in the root directory
Set the backbone feature extraction network and network weights to be used in eval LFW.py.
Run eval-LFW.py to evaluate the accuracy of the model.






