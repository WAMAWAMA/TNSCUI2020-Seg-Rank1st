# 1st place solution in MICCAI 2020 TN-SCUI challenge 
This is the source code of the 1st place solution for segmentation task (with IoU 82.54%) in 2020 TN-SCUI challenge.

[[Challenge leaderboard](https://tn-scui2020.grand-challenge.org/evaluation/leaderboard/)]



## Pipeline of our solution
We use a simple cascaded framework for segmenting nodules, it can be easily extended to other single-target segmentation tasks.

Our pipeline is shown in the figure below.
![something](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/pic/%E5%88%86%E5%89%B2%E8%AE%AD%E7%BB%83%E6%B5%8B%E8%AF%95%E8%BF%87%E7%A8%8B.svg)

<details>
<summary>Click here to view more details of method</summary>
 
## Method
### Data preprocessing
Due to different acquisition protocols, some thyroid ultrasound images have irrelevant regions (as shown in the first Figure). First, we remove these regions which may bring redundant features by using a threshold-based approach. Specifically, we perform the operation of averaging along the x and y axes on original images with pixel values from 0 to 255, respectively, after which rows and columns with mean values less than 5 are removed. Then the processed images are resized to 256×256 pixels as the input of the first segmentation network.

### Cascaded segmentation framework
We train two networks which share the same encoder-decoder structure with Dice loss function. In practice, we choose **`DeeplabV3+ with efficientnet-B6 encoder`** as the first network and the second network. The first segmentation network (at stage Ⅰ of cascade) is trained to provide the rough localization of nodules, and the second segmentation network (at stage Ⅱ of cascade) is trained for fine segmentation based on the rough localization.Our preliminary experiments show that the provided context information in first network may do not play a significant auxiliary role for refinement of the second network. Therefore, we only train the second network using images within region of interest (ROI) obtained from ground truth (GT). (The input data is the only difference in the process of training the two networks.)
![something](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/pic/%E5%A4%A7%E5%B0%8F%E7%BB%93%E8%8A%82%E5%AF%B9%E6%AF%94.svg)
When training the second network, we expand the nodule ROI obtained from GT, then the image in the expanded ROI is cropped out and resized to 512×512 pixels for feeding the second network. We observe that, in most cases, the large nodule generally has a clear boundary, and the gray value of small nodule is quite different from that of surrounding normal thyroid tissue (as shown in the above figure). Therefore, background information (the tissue around the nodule) is significant for segmenting small nodules. As shown in Figure below, in the preprocessed image with the size of 256×256 pixels, the minimum external square of the nodule ROI is obtained first, and then the external expansion m is set to 20 if the edge length n of the square is greater than 80, otherwise the m is set to 30. 
![something](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/pic/%E5%88%86%E5%89%B2%E9%A2%84%E5%A4%84%E7%90%86%E8%BF%87%E7%A8%8B.svg)


### Data augmentation and test time augmentation
In both two task, following methods are performed in data augmentation: 1) horizontal flipping, 2) vertical flipping, 3) random cropping, 4) random affine transformation, 5) random scaling, 6) random translation, 7) random rotation, and 8) random shearing transformation. In addition, one of the following methods was randomly selected for additional augmentation: 1) sharpening, 2) local distortion, 3) adjustment of contrast, 4) blurring (Gaussian, mean, median), 5) addition of Gaussian noise, and 6) erasing. 

Test time augmentation (TTA) generally improves the generalization ability of the segmentation model. In our framework, the TTA includes vertical flipping, horizontal flipping, and rotation of 180 degrees for the segmentation task.

### Cross validation with a size and category balance strategy
5-fold cross validation is used to evaluate the performance of our proposed method. In our opinion, it is necessary to keep the size and category distribution of nodules similar in the training and validation sets. In practice, the size of a nodule is the number of pixels of the nodule after unifying preprocessed image to 256×256 pixels. We stratified the size into three grades: 1) less than 1722 pixels, 2) less than 5666 pixels and greater than 1722 pixels, and 3) greater than 5666 pixels. These two thresholds, 1722 pixels and 5666 pixels, were close to the tertiles, and the size stratification was statistically significantly associated with the benign and malignant categories by the chisquare test (p<0.01). We divided images in each size grade group into 5 folds and combined different grades of single fold into new single fold. This strategy ensured that final 5 folds had similar size and category distributions.

</details>
 
**In summary, what we do in our solution are:**
 - preprocessing to remove irrelevant regions
 - using a cascaded framework
 - 5-fold cross-validation (CV) strategy with a balanced nodule size and category distribution
 - using test time augmentation (TTA)
 - model ensembling: since we trained two networks separately in 5-fold CV , we combined any one first network and one second network as a pair, and finally we got 25 pairs (or inference results). We use [`step4_Merge.py`](https://github.com/WAMAWAMA/TN_SCUI_test/blob/master/step2to4_train_validate_inference/step4_Merge.py) to merge 25 inference results into a final ensemble result by pixel-wised voting

## Segmentation results on 2020 TN-SCUI training dataset and DDTI dataset
We test our method on 2020 TN-SCUI training dataset(with 3644 images or nodules, malignant 2003 : benign 1641). The segmentation results of 5-fold CV based on "DeeplabV3+ with efficientnet-B6 encoder" are as following:
|fold|Stage Ⅰ|TTA at stage Ⅰ|Stage Ⅱ|TTA at stage Ⅱ|DsC|IoU (%)|
|-------------|:-:|:-:|:-:|:-:|:--:|:--:|
| 1      |√  |   |   |   |0.8699|79.00|
| 1      |√  |√  |   |   |0.8775|80.01|
| 1      |√  |   |√  |   |0.8814|80.75|
| 1      |√  |√  |√  |   |0.8841|81.05|
| 1      |√  |   |√  |√  |0.8840|81.16|
| 1      |√  |√  |√  |√  |0.8864|81.44|
| 2      |√  |√  |√  |√  |0.8900|81.99|
| 3      |√  |√  |√  |√  |0.8827|81.07|
| 4      |√  |√  |√  |√  |0.8803|80.56|
| 5      |√  |√  |√  |√  |0.8917|82.07|

<details>
<summary>Click here to view complete TNSCUI segmentation results </summary>
 
|fold|Stage Ⅰ|TTA at stage Ⅰ|Stage Ⅱ|TTA at stage Ⅱ|DsC|IoU (%)|
|-------------|:-:|:-:|:-:|:-:|:--:|:--:|
| 1      |√  |   |   |   |0.8699|79.00|
| 1      |√  |√  |   |   |0.8775|80.01|
| 1      |√  |   |√  |   |0.8814|80.75|
| 1      |√  |√  |√  |   |0.8841|81.05|
| 1      |√  |   |√  |√  |0.8840|81.16|
| 1      |√  |√  |√  |√  |0.8864|81.44|
| 2      |√  |   |   |   |0.8780|80.16|
| 2      |√  |√  |   |   |0.8825|80.80|
| 2      |√  |   |√  |   |0.8872|81.52|
| 2      |√  |√  |√  |   |0.8873|81.56|
| 2      |√  |   |√  |√  |0.8894|81.91|
| 2      |√  |√  |√  |√  |0.8900|81.99|
| 3      |√  |   |   |   |0.8612|78.22|
| 3      |√  |√  |   |   |0.8744|79.77|
| 3      |√  |   |√  |   |0.8710|79.59|
| 3      |√  |√  |√  |   |0.8808|80.66|
| 3      |√  |   |√  |√  |0.8753|80.30|
| 3      |√  |√  |√  |√  |0.8827|81.07|
| 4      |√  |   |   |   |0.8664|78.53|
| 4      |√  |√  |   |   |0.8742|79.44|
| 4      |√  |   |√  |   |0.8742|79.80|
| 4      |√  |√  |√  |   |0.8777|80.12|
| 4      |√  |   |√  |√  |0.8771|80.27|
| 4      |√  |√  |√  |√  |0.8803|80.56|
| 5      |√  |   |   |   |0.8820|80.44|
| 5      |√  |√  |   |   |0.8874|81.22|
| 5      |√  |   |√  |   |0.8869|81.38|
| 5      |√  |√  |√  |   |0.8871|81.37|
| 5      |√  |   |√  |√  |0.8913|82.05|
| 5      |√  |√  |√  |√  |0.8917|82.07|

</details>


We also test our method on another open access thyroid nodule ultrasound image dataset 
(DDTI public database [[https://doi.org/10.1117/12.2073532](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9287/1/An-open-access-thyroid-ultrasound-image-database/10.1117/12.2073532.short)], 
with 637 images or nodules after our preprocessing and data cleaning). The segmentation results of 5-fold CV based on "DeeplabV3+ with efficientnet-B6 encoder" are as following:
|fold|Stage Ⅰ|TTA at stage Ⅰ|Stage Ⅱ|TTA at stage Ⅱ|DsC|IoU (%)|
|-------------|:-:|:-:|:-:|:-:|:--:|:--:|
| 1      |√  |   |   |   |0.8391|74.80|
| 1      |√  |√  |   |   |0.8435|75.49|
| 1      |√  |   |√  |   |0.8380|74.79|
| 1      |√  |√  |√  |   |0.8406|75.16|
| 1      |√  |   |√  |√  |0.8440|75.77|
| 1      |√  |√  |√  |√  |0.8469|76.17|
| 2      |√  |√  |√  |√  |0.8392|74.57|
| 3      |√  |√  |√  |√  |0.8756|79.61|
| 4      |√  |√  |√  |√  |0.8131|72.48|
| 5      |√  |√  |√  |√  |0.8576|77.74|

<details>
<summary>Click here to view complete DDTI segmentation results </summary>
 
|fold|Stage Ⅰ|TTA at stage Ⅰ|Stage Ⅱ|TTA at stage Ⅱ|DsC|IoU (%)|
|-------------|:-:|:-:|:-:|:-:|:--:|:--:|
| 1      |√  |   |   |   |0.8391|74.80|
| 1      |√  |√  |   |   |0.8435|75.49|
| 1      |√  |   |√  |   |0.8380|74.79|
| 1      |√  |√  |√  |   |0.8406|75.16|
| 1      |√  |   |√  |√  |0.8440|75.77|
| 1      |√  |√  |√  |√  |0.8469|76.17|
| 2      |√  |   |   |   |0.8242|72.23|
| 2      |√  |√  |   |   |0.8295|72.99|
| 2      |√  |   |√  |   |0.8373|74.30|
| 2      |√  |√  |√  |   |0.8373|74.33|
| 2      |√  |   |√  |√  |0.8373|74.71|
| 2      |√  |√  |√  |√  |0.8373|74.57|
| 3      |√  |   |   |   |0.8588|77.15|
| 3      |√  |√  |   |   |0.8672|78.26|
| 3      |√  |   |√  |   |0.8663|78.42|
| 3      |√  |√  |√  |   |0.8717|78.95|
| 3      |√  |   |√  |√  |0.8712|79.16|
| 3      |√  |√  |√  |√  |0.8756|79.61|
| 4      |√  |   |   |   |0.8081|71.27|
| 4      |√  |√  |   |   |0.8110|72.05|
| 4      |√  |   |√  |   |0.8037|70.85|
| 4      |√  |√  |√  |   |0.8004|70.79|
| 4      |√  |   |√  |√  |0.8151|72.35|
| 4      |√  |√  |√  |√  |0.8131|72.48|
| 5      |√  |   |   |   |0.8445|75.45|
| 5      |√  |√  |   |   |0.8523|76.47|
| 5      |√  |   |√  |   |0.8495|76.53|
| 5      |√  |√  |√  |   |0.8528|77.06|
| 5      |√  |   |√  |√  |0.8535|77.19|
| 5      |√  |√  |√  |√  |0.8576|77.74|

</details>


## Reproducing the experiment results
You can follow the step-by-step tutorial below, or just download the pretrained weights and run 
[`test_fold1.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/test_fold1.py)
to reproduce our results. (note that we only provide the weights of 1st fold in 5-fold CV). 
 - The pretrained weights (DeeplabV3+ with efficientnet-B6 encoder) at stage Ⅰ and Ⅱ trained on TN-SCUI and DDTI dataset of fold1 are available at 
 [[GoogleDrive](https://drive.google.com/file/d/16CldjuNztEZp2E3SJzXTHp--JggiNCMX/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1epyR5xOBBF4rGLQdHm9NhA), password:`qxds`]
 - The predicted segmentation masks of TN-SCUI final testing dataset (final submission with IoU 82.45%) are available at 
 [[GoogleDrive](https://drive.google.com/file/d/1J-PNbzgO5R8Jnx62uOOk3nCkdo1TqElJ/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1GlgfzHB9RgETHOXbcm5zIg), password:`qxds`]
 - We have processed the DDTI dataset into the format of 2020 TN-SCUI dataset，you can download it from 
 [[GoogleDrive](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1E-28rkg94Jc8NLyKhe2q3g), password:`qxds`]

    
 ## Statement
We consulted the official organizer of 2020 TN-SCUI via email, and the official reply is that using of external dataset is not allowed in the 2020 TN-SCUI challenge, and participants are not allowed to provide the challenge data anywhere. **Therefore, we hereby declare that we did not use any external datasets (such as the DDTI dataset) in the competition, and will not provide TNSCUI data anywhere.** The DDTI dataset processed by us is only used as demo data for this code, so that everyone can run the entire scripts with a ready-made data.
   
# How to run this code on the TN-SCUI dataset or your own custom dataset?
This code can be easily performed on single-target segmentation tasks. Here, we split the whole process into **5 steps** so that you can easily replicate our results or perform the whole pipeline on your private custom dataset. 
 - step0, preparation of environment
 - step1, run the script [`step1_preprocessing.m`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step1_preprocessing/step1_preprocessing.m) to perform the preprocessing
 - step2, run the script [`step2_TrainAndValidate.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step2_TrainAndValidate.py) to train and validate the CNN model
 - step3, run the script [`step3_TestOrInference.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step3_TestOrInference.py) to testing model on original unprocessed images
 - step4 (optional), run the script [`step4_Merge.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step4_Merge.py) to get ensemble predicted result


**You should prepare your data in the format of 2020 TN-SCUI dataset, this is very simple, you only need to prepare:**
 - two folders store PNG format images and masks respectively. Note that the file names (ID) of the images and masks should be integers and correspond to each other
 - a csv file that stores the image names (ID) and category. If there is only a segmentation task without a classification task, you can just falsify the category. For example, half of the data randomly is positive , and the other half is negative

Since the sponsor of TNSCUI challenge does not allow the dissemination of the 2020 TN-SCUI dataset, we give another thyroid ultrasound dataset demo for your reference. The demo dataset was derived from the DDTI open access dataset, and we have processed the dataset into the format of 2020 TN-SCUI dataset，you can download it from 
[[GoogleDrive](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view?usp=sharing)]
[[BaiduWP](https://pan.baidu.com/s/1E-28rkg94Jc8NLyKhe2q3g), password:`qxds`].


## Step0 preparation of environment
We have tested our code in following environment：
 - [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) == 0.1.0 (with install command `pip install segmentation-models-pytorch`)
 - [`ttach`](https://github.com/qubvel/ttach) == 0.0.3
 - `torch` >＝1.0.0
 - `torchvision`
 - [`imgaug`](https://github.com/aleju/imgaug)

For installing `segmentation_models_pytorch (smp)`, there are two commands：
```linux
pip install segmentation-models-pytorch
```
Use the above command, you can install the released version that can run with `torch` <= 1.1.0, but without some decoders such as deeplabV3, deeplabV3+ and PAN.

```linux
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
Use the above command, you will install the latest version which includes decoders such as deeplabV3, deeplabV3+ and PAN, but requiring `torch` >= 1.2.0.

We modified the original code of `smp`, to make it can be run with `torch` <= 1.1.0, and also have the latest version of the `smp` library encoder such as deeplabV3, deeplabV3+ and PAN. The **modified `smp`** is in the filefolder [`segmentation_models_pytorch_4TorchLessThan120`](https://github.com/WAMAWAMA/TN_SCUI_test/tree/master/step2to4_train_validate_inference/segmentation_models_pytorch_4TorchLessThan120), 
here are two ways to use it:
1. install `smp` with `pip install segmentation-models-pytorch` firstly, then just copy the `segmentation_models_pytorch_4TorchLessThan120` into your project folder to use it
2. copy the `segmentation_models_pytorch_4TorchLessThan120` into your project folder, then install these libs: `torchvision>`=0.3.0, `pretrainedmodels`==0.7.4, `efficientnet-pytorch`>=0.5.1



## Step1 preprocessing
In step1, you should run the script [`step1_preprocessing.m`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step1_preprocessing/step1_preprocessing.m) 
in [**MATLAB**](https://www.mathworks.com/products/matlab.html)
to perform the preprocessing. For both TN-SCUI and DDTI dataset, there are some irrelevant regions to be removed. Because we used a cross-validation with a nodule size and category balance strategy, we should get the size of nodules for cross-validation.

After the code runs, you will get two folders and a csv file:
 - the two filefolders are named `stage1` and `stage2`, the data in folder `stage1` is used to train the first network, which contains the preprocessed image with no irrelevant regions; and `stage2` is used to train the second network, which contains the image in the expanded ROI
 - the csv file named `train.csv` with below dataformat, the size of a nodule is the number of pixels of the nodule after unifying preprocessed image to 256×256 pixels

| id       |cate| size |
|:--------:|:--:|:----:|
| 001.PNG  |0   | 2882 |
| 001.PNG  |1   | 3277 |
| 001.PNG  |1   | 3222 |
| 001.PNG  |1   | 256  |
| ┋        |┋   | ┋    |

**The example of preprocessed DDTI dataset was given here
 [[GoogleDrive](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1E-28rkg94Jc8NLyKhe2q3g), password:`qxds`]**

## Step2 training and validatation
### For training phase：
In step2, you should run the script 
[`step2_TrainAndValidate.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step2_TrainAndValidate.py)
to train the first or second network in the cascaded framework (you need to train the two networks separately), **that is, the training process (such as hyperparameters) is the same for both first and second network except for the different training datasets used.** With the csv file `train.csv`, you can directly perform K-fold cross validation (default is 5-fold), and the script uses a fixed random seed to ensure that the K-fold cv of each experiment is repeatable (please note that the same random seed may cause different K-fold under different versions of `scikit-learn`).

For example, you can train the first fold of the first network in two ways: 
1. modify (set) following parameters in the script `step2TrainAndValidate.py` and run the script.
```python
""" just set your param as the default value """
# hyperparameters
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--batch_size_test', type=int, default=30)
# custom parameters
parser.add_argument('--Task_name', type=str, default='dpv3plus_stage1_', help='DIR name,Task name')
parser.add_argument('--csv_file', type=str, default='./DDTI/2_preprocessed_data/train.csv')
parser.add_argument('--filepath_img', type=str, default='./DDTI/2_preprocessed_data/stage1/p_image')
parser.add_argument('--filepath_mask', type=str, default='./DDTI/2_preprocessed_data/stage1/p_image')
parser.add_argument('--fold_K', type=int, default=5, help='number of cross-validation folds')
parser.add_argument('--fold_idx', type=int, default=1)
```
2. use the following command at the terminal to transmit parameters and run the script.
```
python step2_TrainAndValidate.py \
--batch_size=3 \
--Task_name='dpv3plus_stage1_'\
--csv_file='./DDTI/2_preprocessed_data/train.csv' \
--filepath_img='./DDTI/2_preprocessed_data/stage1/p_image' \
--filepath_mask='./DDTI/2_preprocessed_data/stage1/p_mask' \
--fold_idx=1
```

*Something else in training process:*
 - when the GPU is geforce 1080ti, the training batchsize can be set as 10 for first network and 3 for second network, and testing batchsize can be set as 30 for both network
 - the weights with the highest IoU on the validation set will be saved
 - by specifying the parameter `fold_idx` from 1 to K and running repeatedly, you can complete the training of all K folds. The results will be saved in the `result` folder under the project. You can view the training log in real time through `tensorboardX`
 - for training the second network, you only need to change the path of the training data (`filepath_img` and `filepath_mask`) and `Task_name`
 - we do not validation set during training, actually only use the training set and the test set, and copy the test set to the validation set. If you need to use a validation set, you can specify the parameter `validate_flag` as True in the script `step2TrainAndValidate.py`
 - if you do not want to perform cross-validation, or have already prepared a train and validate and test set, just modify the code in the funtion [`main()`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/c4a85db3d712a4ac1f223832717e0a4491fbdba2/step2to4_train_validate_inference/step2_TrainAndValidate.py#L36)
with following code:
```python
train_list = get_filelist_frompath(train_img_folder,'PNG') 
train_list_GT = [train_mask_folder+sep+i.split(sep)[-1] for i in train_list]
test_list = get_filelist_frompath(test_img_folder,'PNG') 
test_list_GT = [test_mask_folder+sep+i.split(sep)[-1] for i in test_list]
```


### For validation and testing phase
In fact, there is no need to re-run the script, because the test has been performed during the training process, and the detailed information of the result is recorded in the `record.xlsx` or `record.txt`. But you can also confirm the final training effect by running the script `step2TrainAndValidate.py` after the training by setting the parameter `mode` as `test`.


**It should be noted that the dataset validated in the training phase is preprocessed. Therefore, if you want to perform test based on the original unprocessed images, please refer to step3.**

## Step3 testing (or inference or predicting)
In step3, you should run the script 
[`step3_TestOrInference.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step3_TestOrInference.py)
to perform inference based on the original unprocessed image. 
By using the script `step3_TestOrInference.py`, you can:
 - perform test of cross-validation (based on the original unprocessed images)
 - inference on new data


Here are some parameters should be noticed:
```python
fold_flag = False # False for inference on new data, and True for cross-validation
mask_path = None  # None for inference on new data
save_path = None  # If you don't want to save the inference result, set it to None
c1_size = 256  # The input image size for training the first network
c1_tta = False # Whether to use TTA for the first network
orimg=False    # False for no preprocessing to remove irrelevant areas
use_c2_flag = False # Whether to use the second network
c2_size = 512 # The input image size for training the second network
c2_tta = False # Whether to use TTA for the second network
```

When you get multiple results after inference, you can use the script 
[`step4_Merge.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step4_Merge.py)
to integrate all the results. We use a pixel-wised voting method, and recommend voting on odd numbers of results to avoid fluctuations in making decision.


# Acknowledgement
- Thanks to the organizers of the 2020 TN-SCUI challenge
- Thanks to the open access DDTI dataset for providing ultrasound data and annotation of thyroid nodules
- Thanks to [qubvel](https://github.com/qubvel), the author of `smg` and `ttach`, all network and TTA used in this code come from his implement



