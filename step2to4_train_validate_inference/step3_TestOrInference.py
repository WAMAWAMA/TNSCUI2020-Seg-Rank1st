# 一张一张图片的计算dice,然后保存
# 计算dice不是最终目的,这个脚本要作为最终的图片输出脚本
import torch
import ttach as tta
from PIL import Image
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.transform import resize
from torchvision import transforms as T

import segmentation_models_pytorch_4TorchLessThan120 as smp
from tnscui_utils.TNSCUI_preprocess import TNSCUI_preprocess
from tnscui_utils.TNSUCI_util import *


def getIOU(SR,GT):
    """
    都是二值图
    :param SR: 
    :param GT: 
    :return: 
    """
    TP = (SR+GT==2).astype(np.float32)
    FP = (SR+(1-GT)==2).astype(np.float32)
    FN = ((1-SR)+GT==2).astype(np.float32)

    IOU = float(np.sum(TP))/(float(np.sum(TP+FP+FN)) + 1e-6)

    return IOU

def getDSC(SR,GT):
    """
    都是二值图
    :param SR: 
    :param GT: 
    :return: 
    """


    Inter = np.sum(((SR+GT)==2).astype(np.float32))
    DC = float(2*Inter)/(float(np.sum(SR)+np.sum(GT)) + 1e-6)

    return DC

def largestConnectComponent(bw_img):
    if np.sum(bw_img)==0:
        return bw_img
    labeled_img, num = sklabel(bw_img, neighbors=4, background=0, return_num=True)
    if num == 1:
        return bw_img


    max_label = 0
    max_num = 0
    for i in range(0,num):
        print(i)
        if np.sum(labeled_img == (i+1)) > max_num:
            max_num = np.sum(labeled_img == (i+1))
            max_label = i+1
    mcr = (labeled_img == max_label)
    return mcr.astype(np.int)


def preprocess(mask_c1_array_biggest, c1_size=256):
    if np.sum(mask_c1_array_biggest) == 0:
            minr, minc, maxr, maxc = [0, 0, c1_size, c1_size]
    else:
        region = regionprops(mask_c1_array_biggest)[0]
        minr, minc, maxr, maxc = region.bbox

    dim1_center, dim2_center = [(maxr + minr) // 2, (maxc + minc) // 2]
    max_length = max(maxr - minr, maxc - minc)

    max_lengthl = int((c1_size/256)*80)
    preprocess1 = int((c1_size/256)*19)
    pp22 = int((c1_size/256)*31)


    if max_length > max_lengthl:
        ex_pixel = preprocess1 + max_length // 2
    else:
        ex_pixel = pp22 + max_length // 2

    dim1_cut_min = dim1_center - ex_pixel
    dim1_cut_max = dim1_center + ex_pixel
    dim2_cut_min = dim2_center - ex_pixel
    dim2_cut_max = dim2_center + ex_pixel

    if dim1_cut_min < 0:
        dim1_cut_min = 0
    if dim2_cut_min < 0:
        dim2_cut_min = 0
    if dim1_cut_max > c1_size:
        dim1_cut_max = c1_size
    if dim2_cut_max > c1_size:
        dim2_cut_max = c1_size
    return [dim1_cut_min,dim1_cut_max,dim2_cut_min,dim2_cut_max]


if __name__ == '__main__':
    """处理的逻辑:
    1)读取图片,有mask则读取mask
    2)图片预处理,比如去黑边的操作,保留去黑边的坐标index1
    3)cascade1,使用TTA,预测mask(后续可以加后处理,如cascadePSP)
    4)预测mask的roi,上下左右随机外扩不同像素,相当于一种tta,然后储存tta外扩像素细节到一个list,为index2
    5)cascade2,使用TTA,在抠出图基础上(resize到256)进行预测,之后还原大小,(后续可以加后处理,如cascadePSP)
    6)根据index2,还原分割结果到原位置,再根据index1,还原到最初位置
    7)和mask计算指标
    9)保存预测图片到指定文件夹
    """
    saveas = 'mask' # prob概率图形式保存,mask二值图形式保存
    # 路径设置
    img_path = r'/root/桌面/DDTI/1_or_data/image'
    mask_path = r'/root/桌面/DDTI/1_or_data/mask'
    csv_file = r'/root/桌面/DDTI/2_preprocessed_data/train.csv'
    save_path = None

    # 是否使用设定好的分折信息
    fold_flag = True  # 预测测试集则设置为False直接读取img_path中PNG文件进行测试,True则使用分折信息
    fold_K = 5
    fold_index = 1

    # task_name
    task_name = r' '

    # cascade参数
    weight_c1=r'/LAOAWNG/git/result/DDTI/Deeplabv3plus_ef6_256_cascade1_5_1/models/best_unet_score.pkl'
    weight_c2=r'/LAOAWNG/git/result/DDTI/Deeplabv3plus_ef6_512_cascade2_5_1/models/best_unet_score.pkl'

    c1_size = 256
    c1_tta = False
    orimg=False

    use_c2_flag = False
    c2_tta = False
    c2_size = 512
    c2_resize_order = 0
    # GPU
    cuda_idx = 2


# ===============================================================================================

    # 设置GPU相关
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)



    # ===============
    device = torch.device('cuda')
    # 读取测试集
    if fold_flag:
        _, test = get_fold_filelist(csv_file, K=fold_K, fold=fold_index)
        test_img_list = [img_path+sep+i[0] for i in test]
        if mask_path is not None:
            test_mask_list = [mask_path+sep+i[0] for i in test]
    else:
        test_img_list = get_filelist_frompath(img_path,'PNG')
        if mask_path is not None:
            test_mask_list = [mask_path + sep + i.split(sep)[-1] for i in test_img_list]

    # 构建两个模型
    with torch.no_grad():
        # tta设置
        tta_trans = tta.Compose([
            tta.VerticalFlip(),
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0,180]),
        ])
        # 构建模型
        # cascade1
        model_cascade1 = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1, classes=1)
        model_cascade1.to(device)
        model_cascade1.load_state_dict(torch.load(weight_c1))
        if c1_tta:
            model_cascade1 = tta.SegmentationTTAWrapper(model_cascade1, tta_trans,merge_mode='mean')
        model_cascade1.eval()
        # cascade2
        model_cascade2 = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1, classes=1)
        # model_cascade2 = smp.Unet(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1, classes=1, encoder_depth=5, decoder_attention_type='scse')
        # model_cascade2 = smp.PAN(encoder_name="efficientnet-b6",encoder_weights='imagenet',	in_channels=1, classes=1)
        model_cascade2.to(device)
        model_cascade2.load_state_dict(torch.load(weight_c2))
        if c2_tta:
            model_cascade2 = tta.SegmentationTTAWrapper(model_cascade2, tta_trans,merge_mode='mean')
        model_cascade2.eval()


        # 指标
        IOU_list = []
        DSC_list = []
        ioumin3 = 0

        #
        for index, img_file in enumerate(test_img_list):
            print(task_name,'\n',img_file,index+1,'/',len(test_img_list),' c1:',c1_size,' c2: ', c2_size)
            with torch.no_grad():
                # 读取GT数据
                if mask_path is not None:
                    mask_file = test_mask_list[index]
                    GT = Image.open(mask_file)
                    Transform_GT = T.Compose([T.ToTensor()])
                    GT = Transform_GT(GT)
                    GT_array = (torch.squeeze(GT)).data.cpu().numpy()



                # 处理原图
                img, cut_image_orshape, or_shape, location = TNSCUI_preprocess(img_file,outputsize=c1_size,orimg=orimg)
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)
                img = img.to(device)
                img_array = (torch.squeeze(img)).data.cpu().numpy()

                """过一遍cascade1"""
                # 获取cascade1输出
                with torch.no_grad():
                    mask_c1 = model_cascade1(img)
                    mask_c1 = torch.sigmoid(mask_c1)
                    mask_c1_array = (torch.squeeze(mask_c1)).data.cpu().numpy()
                    mask_c1_array = (mask_c1_array>0.5)
                    mask_c1_array = mask_c1_array.astype(np.float32)
                    # 获取最大联通域
                    mask_c1_array_biggest = largestConnectComponent(mask_c1_array.astype(np.int))
                    print("c1 outputsize:",mask_c1_array_biggest.shape)



                """过一遍cascade2"""
                with torch.no_grad():
                    if use_c2_flag:
                        # 获取roi的bounding box坐标
                        dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max = preprocess(mask_c1_array_biggest,c1_size)
                        # 根据roi的bounding box坐标，获取img区域
                        img_array_roi = img_array[dim1_cut_min:dim1_cut_max,dim2_cut_min:dim2_cut_max]
                        img_array_roi_shape = img_array_roi.shape
                        img_array_roi = resize(img_array_roi, (c2_size, c2_size), order=3)
                        img_array_roi_tensor = torch.tensor(data = img_array_roi,dtype=img.dtype)
                        img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor,0)
                        img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor,0).to(device)
                        # 获取cascade2输出,并还原大小
                        print('use cascade2')
                        mask_c2 = model_cascade2(img_array_roi_tensor)
                        mask_c2 = torch.sigmoid(mask_c2)
                        mask_c2_array = (torch.squeeze(mask_c2)).data.cpu().numpy()
                        if saveas == 'mask':
                            cascade2_t = 0.5
                            mask_c2_array = (mask_c2_array>cascade2_t)
                            print(cascade2_t)
                        mask_c2_array = mask_c2_array.astype(np.float32)
                        mask_c2_array = resize(mask_c2_array, img_array_roi_shape, order=c2_resize_order)
                        # 放回cascade1输出的mask
                        mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                        mask_c1_array_biggest[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max] = mask_c2_array
                        mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                        print(np.unique(mask_c1_array_biggest))
                        print(np.sum(mask_c1_array_biggest))


                # 根据预处理信息，首先还原到原始size，之后放回原图位置
                mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
                final_mask = np.zeros(shape=or_shape, dtype=mask_c1_array_biggest.dtype)
                mask_c1_array_biggest = resize(mask_c1_array_biggest, cut_image_orshape, order=1)
                final_mask[location[0]:location[1],location[2]:location[3]]=mask_c1_array_biggest



                # 变成二值图
                if saveas == 'mask':
                    final_mask = (final_mask > 0.5)
                    # final_mask = (final_mask > 0.05)
                final_mask = final_mask.astype(np.float32)
                print(np.unique(final_mask))
                print(np.sum(final_mask))


                # 如果有GT的话，计算指标
                if mask_path is not None:
                    if getIOU(final_mask, GT_array)<0.3:
                        ioumin3 = ioumin3+1
                    IOU_list.append(getIOU(final_mask, GT_array))
                    print('IOU:',getIOU(final_mask, GT_array))
                    IOU_final = np.mean(IOU_list)
                    print('fold:',fold_index,'  IOU_final',IOU_final)

                    DSC_list.append(getDSC(final_mask, GT_array))
                    print('DSC:',getDSC(final_mask, GT_array))
                    DSC_final = np.mean(DSC_list)
                    print('fold:',fold_index,'  DSC_final',DSC_final)

                # 保存图像
                if save_path is not None:
                    final_mask = final_mask*255
                    final_mask = final_mask.astype(np.uint8)
                    print(np.unique(final_mask),'\n')
                    # print(np.max())
                    final_savepath = save_path+sep+img_file.split(sep)[-1]
                    im = Image.fromarray(final_mask)
                    im.save(final_savepath)



            #
            # plt.subplot(1, 2, 1)
            # plt.imshow(mask_c1_array_biggest,cmap=plt.cm.gray)
            # # plt.imshow(final_mask,cmap=plt.cm.gray)
            # plt.subplot(1, 2, 2)
            # plt.imshow(img_array,cmap=plt.cm.gray)
            # plt.show()


    print((ioumin3))
    input()























































































