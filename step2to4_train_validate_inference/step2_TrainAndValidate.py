import argparse

from torch.backends import cudnn

from loader.data_loader import get_loader, get_loader_difficult
from tnscui_utils.TNSUCI_util import *
from tnscui_utils.solver import Solver as Solver_or


def main(config):
    cudnn.benchmark = True
    config.result_path = os.path.join(config.result_path, config.Task_name+str(config.fold_K)+'_'+str(config.fold_idx))
    print(config.result_path)
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        os.makedirs(config.model_path)
        os.makedirs(config.log_dir)
        os.makedirs(os.path.join(config.result_path,'images'))


    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)


    print(config)
    f = open(os.path.join(config.result_path,'config.txt'),'w')
    for key in config.__dict__:
        print('%s: %s'%(key, config.__getattribute__(key)), file=f)
    f.close()

    if config.validate_flag:
        train, valid, test = get_fold_filelist(config.csv_file, K=config.fold_K, fold=config.fold_idx, validation=True)
    else:
        train, test = get_fold_filelist(config.csv_file, K=config.fold_K, fold=config.fold_idx)
    """
    if u want to use fixed folder as img & mask folder, u can use following code 
    
    train_list = get_filelist_frompath(train_img_folder,'PNG') 
    train_list_GT = [train_mask_folder+sep+i.split(sep)[-1] for i in train_list]
    test_list = get_filelist_frompath(test_img_folder,'PNG') 
    test_list_GT = [test_mask_folder+sep+i.split(sep)[-1] for i in test_list]
    
    """


    train_list = [config.filepath_img+sep+i[0] for i in train]
    train_list_GT = [config.filepath_mask+sep+i[0] for i in train]

    test_list = [config.filepath_img+sep+i[0] for i in test]
    test_list_GT = [config.filepath_mask+sep+i[0] for i in test]

    if config.validate_flag:
        valid_list = [config.filepath_img+sep+i[0] for i in valid]
        valid_list_GT = [config.filepath_mask+sep+i[0] for i in valid]
    else:
        # just copy test as validation,
        # also u can get the real valid_list use the func 'get_fold_filelist' by setting the param 'validation' as True
        valid_list = test_list
        valid_list_GT = test_list_GT



    config.train_list = train_list
    config.test_list = test_list
    config.valid_list = valid_list

    if config.aug_type == 'easy':
        print('augmentation with easy level')
        train_loader = get_loader(seg_list=None,
                                  GT_list = train_list_GT,
                                  image_list=train_list,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  augmentation_prob=config.augmentation_prob,)
    elif config.aug_type == 'difficult':
        print('augmentation with difficult level')
        train_loader = get_loader_difficult(seg_list=None,
                                              GT_list=train_list_GT,
                                              image_list=train_list,
                                              image_size=config.image_size,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers,
                                              mode='train',
                                              augmentation_prob=config.augmentation_prob,)
    else:
        raise('difficult or easy')
    valid_loader = get_loader(seg_list=None,
                              GT_list = valid_list_GT,
                            image_list=valid_list,
                            image_size=config.image_size,
                            batch_size=config.batch_size_test,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.,)

    test_loader = get_loader(seg_list=None,
                             GT_list = test_list_GT,
                            image_list=test_list,
                            image_size=config.image_size,
                            batch_size=config.batch_size_test,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.,)


    solver = Solver_or(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        unet_path = os.path.join(config.model_path, 'best_unet_score.pkl')
        if config.tta_mode:
            print(char_color('@,,@   doing with tta test'))
            acc, SE, SP, PC, DC, IOU = solver.test_tta(mode='test', unet_path=unet_path)
        else:
            acc, SE, SP, PC, DC, IOU = solver.test(mode='test', unet_path = unet_path)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f' % (
            acc, SE, SP, PC, DC, IOU))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)  # 网络输入img的size, 即输入会被强制resize到这个大小

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=405)

    parser.add_argument('--num_epochs_decay', type=int, default=60)  # decay开始的最小epoch数
    parser.add_argument('--decay_ratio', type=float, default=0.01) #0~1,每次decay到1*ratio
    parser.add_argument('--decay_step', type=int, default=60)  # epoch

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--batch_size_test', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=3)

    # 设置学习率
    parser.add_argument('--lr', type=float, default=1e-4)  # 初始or最大学习率(单用lovz且多gpu的时候,lr貌似要大一些才可收敛)
    parser.add_argument('--lr_low', type=float, default=1e-12)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)

    parser.add_argument('--lr_warm_epoch', type=int, default=5)  # warmup的epoch数,一般就是5~20,为0或False则不使用
    parser.add_argument('--lr_cos_epoch', type=int, default=350)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用

    # optimizer param
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam

    parser.add_argument('--augmentation_prob', type=float, default=1.0)  # 扩增几率

    parser.add_argument('--save_model_step', type=int, default=20)
    parser.add_argument('--val_step', type=int, default=1)


    # misc
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--tta_mode', type=bool, default=True) # 是否在训练过程中的validation使用tta
    parser.add_argument('--Task_name', type=str, default='test', help='DIR name,Task name')
    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--DataParallel', type=bool, default=False) ##

    # data-parameters
    parser.add_argument('--filepath_img', type=str, default='/root/桌面/DDTI/1_or_data/image')
    parser.add_argument('--filepath_mask', type=str, default='/root/桌面/DDTI/1_or_data/mask')
    parser.add_argument('--csv_file', type=str, default='/root/桌面/DDTI/2_preprocessed_data/train.csv')
    parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')
    parser.add_argument('--fold_idx', type=int, default=1)

    # result&save
    parser.add_argument('--result_path', type=str, default='./result/TNSCUI')
    parser.add_argument('--save_detail_result', type=bool, default=True)
    parser.add_argument('--save_image', type=bool, default=True) # 训练过程中观察图像和结果

    # more param
    parser.add_argument('--test_flag', type=bool, default=False) # 训练过程中是否测试,不测试会节省很多时间
    parser.add_argument('--validate_flag', type=bool, default=False) # 是否有验证集
    parser.add_argument('--aug_type', type=str, default='difficult', help='difficult or easy') # 训练过程中扩增代码,分为dasheng,shaonan

    config = parser.parse_args()
    main(config)



