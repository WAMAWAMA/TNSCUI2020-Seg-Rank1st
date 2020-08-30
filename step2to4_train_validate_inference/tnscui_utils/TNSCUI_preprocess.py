import torch
from PIL import Image
from skimage.transform import resize
from torchvision import transforms as T

from tnscui_utils.TNSUCI_util import *


def TNSCUI_preprocess(image_path,outputsize =256, orimg=False):
    # image_path = r'/media/root/老王3号/challenge/tnscui2020_train/image/1273.PNG'
    img = Image.open(image_path)
    Transform = T.Compose([T.ToTensor()])
    img_tensor = Transform(img)
    img_dtype = img_tensor.dtype
    img_array_fromtensor = (torch.squeeze(img_tensor)).data.cpu().numpy()

    img_array = np.array(img, dtype=np.float32)



    or_shape = img_array.shape  #原始图片的尺寸


    value_x = np.mean(img, 1) #% 为了去除多余行，即每一列平均
    value_y = np.mean(img, 0) #% 为了去除多余列，即每一行平均

    x_hold_range = list((len(value_x) * np.array([0.24 / 3, 2.2 / 3])).astype(np.int))
    y_hold_range = list((len(value_y) * np.array([0.8 / 3, 1.8 / 3])).astype(np.int))
    # x_hold_range = list((len(value_x) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))
    # y_hold_range = list((len(value_y) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))

    # value_thresold = 0
    value_thresold = 5


    x_cut = np.argwhere((value_x<=value_thresold)==True)
    x_cut_min = list(x_cut[x_cut<=x_hold_range[0]])
    if x_cut_min:
        x_cut_min = max(x_cut_min)
    else:
        x_cut_min = 0

    x_cut_max = list(x_cut[x_cut>=x_hold_range[1]])
    if x_cut_max:
        # print('q')
        x_cut_max = min(x_cut_max)
    else:
        x_cut_max = or_shape[0]


    y_cut = np.argwhere((value_y<=value_thresold)==True)
    y_cut_min = list(y_cut[y_cut<=y_hold_range[0]])
    if y_cut_min:
        y_cut_min = max(y_cut_min)
    else:
        y_cut_min = 0

    y_cut_max = list(y_cut[y_cut>=y_hold_range[1]])
    if y_cut_max:
        # print('q')
        y_cut_max = min(y_cut_max)
    else:
        y_cut_max = or_shape[1]


    if orimg:
        x_cut_max = or_shape[0]
        x_cut_min = 0
        y_cut_max = or_shape[1]
        y_cut_min = 0
    # 截取图像
    cut_image = img_array_fromtensor[x_cut_min:x_cut_max,y_cut_min:y_cut_max]
    cut_image_orshape = cut_image.shape

    cut_image = resize(cut_image, (outputsize, outputsize), order=3)

    cut_image_tensor = torch.tensor(data = cut_image,dtype=img_dtype)

    return [cut_image_tensor, cut_image_orshape,or_shape,[x_cut_min,x_cut_max,y_cut_min,y_cut_max]]

def TNSCUI_preprocess4reesemble(image_path,mask_path,outputsize =256):
    # image_path = r'/media/root/老王3号/challenge/tnscui2020_train/image/1273.PNG'

    # 读取mask
    mask = Image.open(mask_path)
    mask = np.asarray(mask)

    # 读取img
    img = Image.open(image_path)
    Transform = T.Compose([T.ToTensor()])
    img_tensor = Transform(img)
    img_dtype = img_tensor.dtype
    img_array_fromtensor = (torch.squeeze(img_tensor)).data.cpu().numpy()

    img_array = np.array(img, dtype=np.float32)



    or_shape = img_array.shape  #原始图片的尺寸


    value_x = np.mean(img, 1) #% 为了去除多余行，即每一列平均
    value_y = np.mean(img, 0) #% 为了去除多余列，即每一行平均

    x_hold_range = list((len(value_x) * np.array([0.24 / 3, 2.2 / 3])).astype(np.int))
    y_hold_range = list((len(value_y) * np.array([0.8 / 3, 1.8 / 3])).astype(np.int))
    # x_hold_range = list((len(value_x) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))
    # y_hold_range = list((len(value_y) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))

    # value_thresold = 0
    value_thresold = 5


    x_cut = np.argwhere((value_x<=value_thresold)==True)
    x_cut_min = list(x_cut[x_cut<=x_hold_range[0]])
    if x_cut_min:
        x_cut_min = max(x_cut_min)
    else:
        x_cut_min = 0

    x_cut_max = list(x_cut[x_cut>=x_hold_range[1]])
    if x_cut_max:
        # print('q')
        x_cut_max = min(x_cut_max)
    else:
        x_cut_max = or_shape[0]




    y_cut = np.argwhere((value_y<=value_thresold)==True)
    y_cut_min = list(y_cut[y_cut<=y_hold_range[0]])
    if y_cut_min:
        y_cut_min = max(y_cut_min)
    else:
        y_cut_min = 0

    y_cut_max = list(y_cut[y_cut>=y_hold_range[1]])
    if y_cut_max:
        # print('q')
        y_cut_max = min(y_cut_max)
    else:
        y_cut_max = or_shape[1]


    # 截取图像
    cut_image = img_array_fromtensor[x_cut_min:x_cut_max,y_cut_min:y_cut_max]
    mask = mask[x_cut_min:x_cut_max,y_cut_min:y_cut_max]

    cut_image_orshape = cut_image.shape

    cut_image = resize(cut_image, (outputsize, outputsize), order=3)
    mask = resize(mask, (outputsize, outputsize), order=0)

    cut_image_tensor = torch.tensor(data = cut_image,dtype=img_dtype)

    return [cut_image_tensor, mask, cut_image_orshape,or_shape,[x_cut_min,x_cut_max,y_cut_min,y_cut_max]]

# plt.subplot(1, 2, 1)
# plt.imshow(img_array,cmap=plt.cm.gray)
# plt.subplot(1, 2, 2)
# plt.imshow(cut_image,cmap=plt.cm.gray)
# plt.show()



