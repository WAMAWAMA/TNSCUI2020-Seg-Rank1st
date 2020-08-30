import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tnscui_utils.TNSUCI_util import *


def get_pic(file_path):
    GT = Image.open(file_path)
    GT = np.asarray(GT)
    GT = GT.astype(np.float32)
    return GT


merge_path = []

merge_path.append(r'/media/root/s1_fold51_s2_fold51')
merge_path.append(r'/media/root/s1_fold51_s2_fold52')
merge_path.append(r'/media/root/s1_fold51_s2_fold53')
merge_path.append(r'/media/root/s1_fold51_s2_fold54')
merge_path.append(r'/media/root/s1_fold51_s2_fold55')

merge_path.append(r'/media/root/s1_fold52_s2_fold51')
merge_path.append(r'/media/root/s1_fold52_s2_fold52')
merge_path.append(r'/media/root/s1_fold52_s2_fold53')
merge_path.append(r'/media/root/s1_fold52_s2_fold54')
merge_path.append(r'/media/root/s1_fold52_s2_fold55')

merge_path.append(r'/media/root/s1_fold53_s2_fold51')
merge_path.append(r'/media/root/s1_fold53_s2_fold52')
merge_path.append(r'/media/root/s1_fold53_s2_fold53')
merge_path.append(r'/media/root/s1_fold53_s2_fold54')
merge_path.append(r'/media/root/s1_fold53_s2_fold55')

merge_path.append(r'/media/root/s1_fold54_s2_fold51')
merge_path.append(r'/media/root/s1_fold54_s2_fold52')
merge_path.append(r'/media/root/s1_fold54_s2_fold53')
merge_path.append(r'/media/root/s1_fold54_s2_fold54')
merge_path.append(r'/media/root/s1_fold54_s2_fold55')

merge_path.append(r'/media/root/s1_fold55_s2_fold51')
merge_path.append(r'/media/root/s1_fold55_s2_fold52')
merge_path.append(r'/media/root/s1_fold55_s2_fold53')
merge_path.append(r'/media/root/s1_fold55_s2_fold54')
merge_path.append(r'/media/root/s1_fold55_s2_fold55')





save_path = r'/media/root/merge'


if not os.path.exists(save_path):
    os.makedirs(save_path)

mask_list = get_filelist_frompath(merge_path[0],'PNG')


for indd, file in enumerate(mask_list):
    print(indd,file)
    file_name = file.split(sep)[-1]
    pic_list = [get_pic(_path+sep+file_name) for _path in merge_path]
    pic_list_array = np.array(pic_list)
    pic_list_array_mean = np.mean(pic_list_array,0)

    final_mask = (pic_list_array_mean > 0.485*255)
    final_mask = final_mask.astype(np.float32)
    # final_mask = largestConnectComponent(final_mask.astype(np.int))
    final_mask = final_mask.astype(np.uint8)
    final_mask = final_mask*255

    if False:
        plt.subplot(1, 2, 1)
        plt.imshow(pic_list_array_mean,cmap=plt.cm.gray)
        plt.subplot(1, 2, 2)
        plt.imshow(final_mask,cmap=plt.cm.gray)
        plt.show()

    # 保存图像
    if True:
        final_savepath = save_path + sep + file_name
        im = Image.fromarray(final_mask)
        im.save(final_savepath)