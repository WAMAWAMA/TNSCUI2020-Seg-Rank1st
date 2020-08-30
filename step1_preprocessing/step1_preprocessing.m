%% 数据预处理代码
% 输入文件夹是原始数据image和mask文件夹
% 输出文件夹是p_image，p_mask
clc;
close all;
clear;

%% 参数
%简单阈值法卡掉crop黑边
value_thresold = 5;  % 因为是unit8型图像灰度值0到255，所以设置10

% 目标尺寸
stage1_aim_size = 256;
stage2_aim_size = 512;

%% 路径设置 set path
img_dir  = 'C:\DDTI\1_or_data\image';
mask_dir = 'C:\DDTI\1_or_data\mask';
csv_file = 'C:\DDTI\1_or_data\category.csv';
save_dir = 'C:\DDTI\2_preprocessed_data';

%% 构建保存文件夹
mkdir([save_dir,filesep,'stage1',filesep,'p_image']);
mkdir([save_dir,filesep,'stage1',filesep,'p_mask']);
mkdir([save_dir,filesep,'stage2',filesep,'p_image']);
mkdir([save_dir,filesep,'stage2',filesep,'p_mask']);

%% 构建结节size统计结果文件
csv_path = [save_dir,filesep,'train.csv'];
% csv文件添加key
fid=fopen(csv_path,'w');
strrr={'ID','CATE','size'};
fprintf(fid,'%s,%s,%s\n',string(strrr));
fclose(fid);

%% 读取数据获得filename_list
filename_list = get_file_list(img_dir,'PNG');

%% 读取数据获取ID和CATE
csv_data = importdata(csv_file);
csv_id_ = csv_data.textdata(2:end,1);
csv_cate = csv_data.data;
csv_id = [];
for i = 1:length(csv_id_)
   t_ = strsplit(csv_id_{i},'.'); 
   csv_id(end+1) = str2double(t_(1));
end
csv_id = csv_id';

%% 预处理

% 逐个处理，横向、纵向裁剪掉黑边,抠出ROI，外扩
for i = 1:length(filename_list)
    id = strsplit(filename_list{i},'.');
    id = str2double(id(1));
    img = imread([img_dir,filesep,filename_list{i}]);
    mask = imread([mask_dir,filesep,filename_list{i}]);
    % preprocess for stage 1
    [img4stage1,mask4stage1] = preprocess(img, mask, value_thresold, stage1_aim_size);
    [img4stage1_,mask4stage1_] = preprocess(img, mask, value_thresold, stage1_aim_size*2);
    
    % preprocess for stage 2
    [img4stage2,mask4stage2] = cutROIwithExpand(img4stage1_, mask4stage1_, stage2_aim_size);
    
    % 计算size
    nodule_size = sum(sum(mask4stage1));
    
    % 写入csv:id,cate,size
    fid=fopen(csv_path,'a');
    fprintf(fid,'%s,%d,%d\n',filename_list{i},csv_cate(csv_id==id),nodule_size);
    fclose(fid);
    
    
    % 保存图片
    % for stage1
    imwrite(img4stage1,[save_dir,filesep,'stage1',filesep,'p_image',filesep,filename_list{i}]);
    imwrite(mask4stage1,[save_dir,filesep,'stage1',filesep,'p_mask',filesep,filename_list{i}]);
    % for stage2
    imwrite(img4stage2,[save_dir,filesep,'stage2',filesep,'p_image',filesep,filename_list{i}]);
    imwrite(mask4stage2,[save_dir,filesep,'stage2',filesep,'p_mask',filesep,filename_list{i}]);

    % 显示进度
    disp([num2str(i),'|',num2str(length(filename_list))]);
    
end
  
    

   

