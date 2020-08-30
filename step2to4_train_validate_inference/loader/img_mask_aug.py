import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


def data_aug(imgs, masks, segs=None):     # 输入图像和标签,输入输出格式为numpy
    # 标准化格式
    imgs = np.array(imgs)
    masks = np.array(masks).astype(np.uint8)
    if segs is not None:
        segs = np.array(segs).astype(np.uint8)

    # print('imgs shape',imgs.shape)
    # 确定batch数
    # if imgs.shape == (256, 256):
    if len(imgs.shape) == 2: # 判断图像是否是二维的,以此判断是否是一个batch
        batch_num = 1
    else:
        batch_num = 1

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)     # 设定随机函数,50%几率扩增,or
    # sometimes = lambda aug: iaa.Sometimes(0.7, aug)     # 设定随机函数,70%几率扩增

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),    # 50%图像进行水平翻转
            iaa.Flipud(0.5),    # 50%图像做垂直翻转

            sometimes(iaa.Crop(percent=(0, 0.1))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%
            # sometimes(iaa.Crop(percent=(0, 0.2))),  # 对随机的一部分图像做crop操作 crop的幅度为0到20% wang

            sometimes(iaa.Affine(                          # 对一部分图像做仿射变换
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},   # 图像缩放为80%到120%之间
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},     # 平移±20%之间
                rotate=(-45, 45),   # 旋转±45度之间
                shear=(-16, 16),    # 剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],   # 使用最邻近差值或者双线性差值
                cval=(0, 255),
                mode=ia.ALL,   # 边缘填充
            )),

            # 使用下面的0个到5个之间的方法去增强图像
            iaa.SomeOf((0, 5),
                       [
                           # 锐化处理
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)),

                           # 扭曲图像的局部区域
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),

                           # 改变对比度
                           iaa.contrast.LinearContrast((0.75, 1.25), per_channel=0.5),

                           # 用高斯模糊，均值模糊，中值模糊中的一种增强
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # 加入高斯噪声
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # 边缘检测，将检测到的赋值0或者255然后叠在原图上(不确定)
                           # sometimes(iaa.OneOf([
                           #     iaa.EdgeDetect(alpha=(0, 0.7)),
                           #     iaa.DirectedEdgeDetect(
                           #         alpha=(0, 0.7), direction=(0.0, 1.0)
                           #     ),
                           # ])),

                           # 浮雕效果(很奇怪的操作,不确定能不能用)
                           # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # 将1%到10%的像素设置为黑色或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),

                       ],

                       random_order=True    # 随机的顺序把这些操作用在图像上
                       )
        ],
        random_order=True   # 随机的顺序把这些操作用在图像上

    )

    if batch_num > 1:   # 多batch处理
        images_aug = np.zeros(imgs.shape)
        segmaps_aug = np.zeros(masks.shape)
        if segs is not None:
            segmaps_aug_seg = np.zeros(masks.shape)
        for i in range(0, batch_num):
            seq_det = seq.to_deterministic()    # 确定一个数据增强的序列
            images_aug[i, :, :] = seq_det.augment_images(imgs[i, :, :])     # 进行增强
            segmap = ia.SegmentationMapsOnImage(masks[i, :, :], shape=masks[i, :, :].shape)  # 分割标签格式
            segmaps_aug[i, :, :] = seq_det.augment_segmentation_maps(segmap).get_arr().astype(np.uint8)   # 将方法应用在分割标签上，并且转换成np类型
            if segs is not None:
                segmap_seg = ia.SegmentationMapsOnImage(segs[i, :, :], shape=segs[i, :, :].shape)  # 分割标签格式
                segmaps_aug_seg[i, :, :] = seq_det.augment_segmentation_maps(segmap_seg).get_arr().astype(np.uint8)

    else:
        seq_det = seq.to_deterministic()  # 确定一个数据增强的序列
        # print('imgs.shape',imgs.shape)
        images_aug = seq_det.augment_images(imgs)  # 进行增强
        segmaps = ia.SegmentationMapsOnImage(masks, shape=masks.shape)  # 分割标签格式
        segmaps_aug = seq_det.augment_segmentation_maps(segmaps).get_arr().astype(np.uint8)  # 将方法应用在分割标签上，并且转换成np类型
        if segs is not None:
            segmap_seg = ia.SegmentationMapsOnImage(segs, shape=segs.shape)  # 分割标签格式
            segmaps_aug_seg = seq_det.augment_segmentation_maps(segmap_seg).get_arr().astype(np.uint8)


    if  segs is not None:
        return images_aug, segmaps_aug, segmaps_aug_seg
    else:
        return images_aug, segmaps_aug

