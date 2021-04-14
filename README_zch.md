# 在MICCAI 2020 TN-SCUI 挑战中排名第一的解决方案
这是2020年TN-SCUI挑战中分割任务(loU为82.54%)第一名解决方案的源代码。
[挑战排行榜](https://tn-scui2020.grand-challenge.org/evaluation/leaderboard/)




## 解决方案的途径
我们使用一个简单的级联框架来分割结节，它可以很容易地扩展到其他单目标分割任务。

我们的途径如下图所示。
![某事](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/pic/%E5%88%86%E5%89%B2%E8%AE%AD%E7%BB%83%E6%B5%8B%E8%AF%95%E8%BF%87%E7%A8%8B.svg)

<details>
<summary>点击这里查看方法的更多细节 </summary>
 
## 方法
### 数据预处理
由于不同的采集协议，部分甲状腺超声图像有不相关区域(如图1所示)。首先，我们采用阈值法去除这些可能带来冗余特征的区域。特别地，我们对像素值从0到255的原始图像进行沿x轴和y轴平均的操作，分别去除均值小于5的行和列。然后将处理后的图像大小调整为256x256像素作为第一个分割网络的输入。

### 级联分割框架
利用骰子损耗函数对两个具有相同编码解码器结构的网络进行训练。事实上，我们选择`DeeplabV3+ with efficientnet-B6 encoder`作为第一网络和第二网络。训练第一个分割网络（级联的I阶段）提供结节的粗略定位，在粗定位的基础上，训练第二次分割网络（级联的II阶段）进行精细分割。我们的初步实验表明，在第一个网络中提供的上下文信息可能不会对第二个网络的细化起到显著的辅助作用。因此，我们只使用ground truth(GT)获得的感兴趣区域（region of interest,ROI）内的图像来训练第二个网络。(在训练这两个网络的过程中，输入数据是唯一的区别。)
![某事](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/pic/%E5%A4%A7%E5%B0%8F%E7%BB%93%E8%8A%82%E5%AF%B9%E6%AF%94.svg)
当训练第二个网络时，我们将GT得到的结节ROI展开，然后将扩大的ROI中的图像裁剪出来，并将其大小调整为512x512像素，以供第二个网络使用。我们观察到，在大多数病例中，大结节一般边界清楚，而且小结节的灰度值与周围正常的甲状腺组织的灰度值差异较大（如图所示）。因此，背景信息（结节周围组织）对于小结节的分割具有重要意义。如下图所示，在预处理后大小为256x256像素的图像中，首先得到结节感兴趣区域的最小外平方，然后若正方形边长n大于80，则外扩m为20，否则m为30。
![某事](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/mater/pic/%E5%88%86%E5%89%B2%E9%A2%84%E5%A4%84%E7%90%86%E8%BF%87%E7%A8%8B.svg)


### 数据增强和测试时间增强
在这两个任务中，以下方法在数据增强中执行： 1) 水平翻转， 2) 垂直翻转， 3) 随机剪切， 4)随机仿射变换 ，5) 随机检尺，6) 随机翻译， 7) 随机旋转, 和 8) 随机剪切变换。 此外，随机选择以下方法之一进行额外的增强： 1) 锐磨， 2)局部辨析， 3) 调整对比， 4) 模糊（高斯，平均值，中值）， 5) 高斯噪声的加法，和 6)擦除。 

TTA（Test time augmentation）通常提高分割模型的泛化能力。在我们的框架中，TTA包括用于分割任务的垂直翻转，水平翻转和180度旋转。

### 通过规模和类别平衡策略进行交叉验证
我们使用五折交叉验证来评估我们所提出的方法的性能。我们的观点是，有必要保持训练集和验证集中结节的大小和类别分布相似。实际上，结节的大小是将预处理后的图像统一为256x256像素后，结节的像素数量。我们将规模分为三个等级：1) 小于1722像素，2) 小于5666像素大于1722像素，和 3)大于5666像素。 这两个阈值，1722像素和5666像素， 很接近三分位数，通过卡方独立检验，大小分层与良、恶性分类有统计学意义（p<0.01）。我们将每个尺寸等级组的图像分为5个等级，并将不同等级的单次折叠合成新的单次折叠。 这一策略确保了最后五次折叠具有类似的大小和种类分布。

</details>
 
**总之，我们在解决方案中所做的是**
 - 预处理去除不相干的区域
 - 实验级联框架
 -五折交叉验证（CV）策略，平衡结节大小和种类分布
 - 使用TTA（Test time augmentation）
 - 模型整体：由于我们在五折CV中分别训练了两个网络，我们将任意一个第一个网络和一个第二个网络组合成一对，最后我们得到25对（或推理结果）我们用[`step4_Merge.py`](https://github.com/WAMAWAMA/TN_SCUI_test/blob/master/step2to4_train_validate_inference/step4_Merge.py) 通过像素投票将25个推断结果合并成最终的集合结果

## 基于2020 TN-SCUI训练数据集和DDTI数据集的分割结果
我们在2020TN-SCUI训练集上测试了我们的方法3644张图像或结节，恶性2003：良性1641)。基于“DeeplabV3+ with efficientnet-B6 encoder”的五折CV分割结果如下：
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
<summary>点击这里查看完整的TNSCUI分割结果 </summary>
 
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


我们也在另一个开源甲状腺结节超声图像数据集上测试了我们的方法 
(DDTI 公共数据库 [[https://doi.org/10.1117/12.2073532](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9287/1/An-open-access-thyroid-ultrasound-image-database/10.1117/12.2073532.short)], 
有637张图像或结节经过我们的预处理和数据清洗)。基于 "DeeplabV3+ with efficientnet-B6 encoder" 的五折CV分割结果如下:
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
<summary>点击这里查看完整的DDTI分割结果</summary>
 
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


## 复现实验结果
你可以按照下面的教程一步一步地练习，也可以下载训练过的举重练习然后运行 
[`test_fold1.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/test_fold1.py)
来复现我们的结果。(注意，我们只提供五折CV中的第一折的权重). 
 - 在I和II阶段预先训练过的权重 (DeeplabV3+ 与 efficientnet-B6 的编码器)，以及在fold1的TN-SCUI和DDTI数据集上训练的权重可以在
 [[GoogleDrive](https://drive.google.com/file/d/16CldjuNztEZp2E3SJzXTHp--JggiNCMX/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1epyR5xOBBF4rGLQdHm9NhA), 密码:`qxds`]获得
 - TN-SCUI最终测试数据集的预测分割掩码(最后提交时 IoU=82.45%) 可以在
 [[GoogleDrive](https://drive.google.com/file/d/1J-PNbzgO5R8Jnx62uOOk3nCkdo1TqElJ/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1GlgfzHB9RgETHOXbcm5zIg),密码:`qxds`]上获得
 -我们已经将DDTI数据集处理成2020TN-SCUI格式的数据，你可以从
 [[GoogleDrive](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1E-28rkg94Jc8NLyKhe2q3g), 密码:`qxds`]上下载

    
 ## 声明
我们通过邮件咨询了2020TN-SCUI的官方组织者，官方的回答是，2020年TN-SCUI的挑战赛不允许使用外部数据集，并且参赛者不允许在任何地方提供挑战数据。**因此，我们在此声明，我们比赛中没有使用任何外部数据集（如DDTI数据集），也不会在任何地方提供TN-SCUI数据** 我们处理的DDTI数据集仅用作这段代码的演示数据，这样每个人都可以用现成的数据运行整个脚本。
   
# 如何在TN-SCUI数据集或您自己定义的数据集上运行此代码呢？
这段代码可以很容易地在单目标分割任务上执行。这里，我们将整个过程分为**五个步骤**，以便您可以轻松复制我们的结果或在您个人自定义数据集上执行整个过程。
 - step0,准备环境
 - step1, 运行脚本[`step1_preprocessing.m`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step1_preprocessing/step1_preprocessing.m) 来执行预处理
 - step2, 运行脚本 [`step2_TrainAndValidate.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step2_TrainAndValidate.py) 训练和验证CNN模型
 - step3,运行脚本[`step3_TestOrInference.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step3_TestOrInference.py) 在未经处理的原始图像上测试模型
 - step4 (可选择的), 运行脚本[`step4_Merge.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step4_Merge.py) 得到集合预测结果


**你应该准备2020 TN-SCUI数据集格式的数据, 这是很简单的, 你只需要准备:**
 - 两个文件夹分别存放PNG格式的图片和掩码。请注意，图像和掩码的文件名(ID)应该是整数，并且彼此对应
 - 存储图像名称(ID)和类别的csv文件。如果只有分割任务而没有分类任务，你可以伪造分类。例如，一半的随机数据是正面的，另一半是负面的

由于TNSCUI挑战赛主办方不允许传播2020TN-SCUI数据集，我们提供另一个甲状腺超声数据集演示供您参考。demo数据集来源于DDTI开源数据集，我们已经将数据集处理为2020TN-SCUI数据集的格式，你可以从
[[GoogleDrive](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view?usp=sharing)]
[[BaiduWP](https://pan.baidu.com/s/1E-28rkg94Jc8NLyKhe2q3g), 密码:`qxds`].上下载


## Step0 准备环境
我们已经在以下环境中测试了代码：
 - [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) == 0.1.0 (安装命令 `pip install segmentation-models-pytorch`)
 - [`ttach`](https://github.com/qubvel/ttach) == 0.0.3
 - `torch` >＝1.0.0
 - `torchvision`
 - [`imgaug`](https://github.com/aleju/imgaug)

要安装`segmentation_models_pytorch (smp)`, 有两个命令：
```linux
pip install segmentation-models-pytorch
```
使用上面的命令， 你可以安装发行的版本，在 `torch` <= 1.1.0时运行，但是没有一些解释器，比如deeplabV3, deeplabV3+ 和 PAN。

```linux
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
使用上述命令，您将安装最新版本，其中包括deeplabV3, deeplabV3+和PAN等解码器，但需要' torch ' >= 1.2.0。

我们修改了原始代码 `smp`, 使它可以在 `torch` <= 1.1.0的环境下运行, 还有最新版本的 `smp` 库编码器如 deeplabV3, deeplabV3+ 和 PAN。 **modified `smp`** 在文件夹[`segmentation_models_pytorch_4TorchLessThan120`](https://github.com/WAMAWAMA/TN_SCUI_test/tree/master/step2to4_train_validate_inference/segmentation_models_pytorch_4TorchLessThan120)中，
这里有两种用法：
1. 首先，用 `pip install segmentation-models-pytorch`安装 `smp`，然后复制 `segmentation_models_pytorch_4TorchLessThan120` 到你的项目文件夹中使用它
2.复制 `segmentation_models_pytorch_4TorchLessThan120` 到你的项目文件夹，然后安装这些libs `torchvision>`=0.3.0, `pretrainedmodels`==0.7.4，`efficientnet-pytorch`>=0.5.1



## Step1 预处理
在步骤1中，您应该在 [**MATLAB**](https://www.mathworks.com/products/matlab.html)中运行脚本 [`step1_preprocessing.m`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step1_preprocessing/step1_preprocessing.m) 

来执行预处理。 对于TN-SCUI和DDTI数据集，都有一些不相关的区域需要删除。因为我们使用了结核大小和类别平衡策略的交叉验证，我们应该得到结核大小进行交叉验证。

代码运行后，你会得到两个文件夹和一个cvs文件：
 - 这两个文件夹名为 `stage1` 和 `stage2`, 文件夹`stage1`中的数据用于训练第一个网络，该网络包含预处理后的图像，没有不相关区域； `stage2`用于训练第二个网络，该网络包含扩展ROI的图像
 - 名为 `train.csv`的cvs文件是下面的这种数据格式,结节的大小是将预处理后的图像统一为256x256像素后，结节的像素数。

| id       |cate| size |
|:--------:|:--:|:----:|
| 001.PNG  |0   | 2882 |
| 001.PNG  |1   | 3277 |
| 001.PNG  |1   | 3222 |
| 001.PNG  |1   | 256  |
| ┋        |┋   | ┋    |

**给出了DDTI数据集的预处理实例
 [[GoogleDrive](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view?usp=sharing)]
 [[BaiduWP](https://pan.baidu.com/s/1E-28rkg94Jc8NLyKhe2q3g), 密码:`qxds`]**

## Step2 训练和验证
### 训练阶段：
在步骤2中， 你应该在级联框架中运行脚本
[`step2_TrainAndValidate.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step2_TrainAndValidate.py)
来训练第一或者第二网络、(你需要分别训练这两个网络)，**也就是说，除了实验不同的训练数据集外，第一和第二网络的训练过程（如超参数）是相同的**。使用cvs文件 `train.csv`, 你可以执行K折交叉验证（默认为五折），脚本使用固定的随机种子，确保每个实验的K折cv是可重复的 (请注意，相同的随机种子在不同版本的`scikit-learn`下可能会产生不同的K折).

例如，你可以用两种方法来训练第一个网络和第一折交叉验证： 
1. 在脚本 `step2TrainAndValidate.py`中修改（设置）以下参数然后运行脚本。
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
2. 在终端上使用以下命令传输参数并运行脚本。
```
python step2_TrainAndValidate.py \
--batch_size=3 \
--Task_name='dpv3plus_stage1_'\
--csv_file='./DDTI/2_preprocessed_data/train.csv' \
--filepath_img='./DDTI/2_preprocessed_data/stage1/p_image' \
--filepath_mask='./DDTI/2_preprocessed_data/stage1/p_image' \
--fold_idx=1
```

*训练过程中的其他情况*
 - 当GPU为geforce 1080ti时，第一个网络的训练批大小可以设置为10，第二个网络可以设置为3，两个网络的测试批大小都可以设置为30
 - 验证集中loU最高的权重将会被保留
 - 通过指定1到k的参数 `fold_idx` 并反复运行， 你可以完成所有k次折叠的实验。结果将保存在项目下的 `result` 文件夹中。你可以通过`tensorboardX`实时查看训练日志
 - 为了训练第二个网络，你只需要更改训练数据的路径 (`filepath_img` 和 `filepath_mask`) 和 `Task_name`
 -我们在训练时不使用验证集，实际上只使用训练集和测试集， 并将测试集复制到验证集。如果需要使用验证集，你可以在脚本`step2TrainAndValidate.py`中将参数 `validate_flag` 指定为True 
 - 如果你不想执行交叉验证，或者已经准备好了训练、验证和测试集，只需用以下代码修改函数 [`main()`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/c4a85db3d712a4ac1f223832717e0a4491fbdba2/step2to4_train_validate_inference/step2_TrainAndValidate.py#L36)中的代码
```python
train_list = get_filelist_frompath(train_img_folder,'PNG') 
train_list_GT = [train_mask_folder+sep+i.split(sep)[-1] for i in train_list]
test_list = get_filelist_frompath(test_img_folder,'PNG') 
test_list_GT = [test_mask_folder+sep+i.split(sep)[-1] for i in test_list]
```


### 验证和测试阶段
事实上，没有必要重新运行脚本，因为测试已经在训练过程中执行，并且结果的详细信息记录在 `record.xlsx` 或 `record.txt`中。通过将参数 `mode` 设置为 `test`后，你也可以通过运行脚本 `step2TrainAndValidate.py` 来确认最终的训练效果。


**需要注意的是，在训练阶段验证的数据集是预处理的。因此，如果你想对未经处理的原始图像进行测试，请参考步骤3**

## Step3 测试（或推理或预测）
在步骤3中，你应该运行脚本
[`step3_TestOrInference.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step3_TestOrInference.py)
基于原始未处理图像进行推理。 
通过运行脚本 `step3_TestOrInference.py`，你可以：
 -执行交叉验证测试（基于未处理的图像）
 - 新数据推理


以下是一些需要注意的参数
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

当你在推断后得到多个结果时，你可以使用脚本 
[`step4_Merge.py`](https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st/blob/master/step2to4_train_validate_inference/step4_Merge.py)
来整合所有的结果。 我们使用像素明智的投票方法，并建议对奇数的结果投票，以避免在决策中的波动。

# 致谢
- 感谢2020 TN-SCUI挑战赛中的组织者
- 感谢开源DDTI数据集提供了超声数据和甲状腺结节注释
- 感谢 [qubvel](https://github.com/qubvel)， `smg` 和 `ttach`的作者，本代码中使用的所有网络和TTA都来自他的实现


