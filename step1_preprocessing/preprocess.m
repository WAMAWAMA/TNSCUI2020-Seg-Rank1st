function [img_p, mask_p] = preprocess(img,mask,value_thresold,stage1_aim_size)
    img_size = size(img);
    %图片因为是unit8的，所以范围是0到255，保存的时候
    
    value_y = mean(img,1); %为了去除多余列，即每一行平均
    value_x = mean(img,2); %为了去除多余行，即每一列平均
    
    % 图片中间不去除，即0.8/3到2.2/3的地方不去除，作为保留部分hold(PS:不同机器需要设定的范围不同)
    x_hold_range = round(length(value_x)*[0.24/3, 2.2/3]);
    y_hold_range = round(length(value_y)*[0.8/3, 1.8/3]);
    
    % 寻找需要去除的行，即0到x_cut_min，和x_cut_max之后的行需要去除
    x_cut = find(value_x<=value_thresold);
    x_cut_min = max(x_cut(x_cut<=x_hold_range(1)));
    x_cut_max = min(x_cut(x_cut>=x_hold_range(2)));
    
    if isempty(x_cut_min)
        x_cut_min = 1;
    end
    if isempty(x_cut_max)
        x_cut_max = img_size(1);
    end
    
    % 寻找需要去除的列，即0到y_cut_min，和y_cut_max之后的列需要去除
    y_cut = find(value_y<=value_thresold);
    y_cut_min = max(y_cut(y_cut<=y_hold_range(1)));
    y_cut_max = min(y_cut(y_cut>=y_hold_range(2)));
   
    if isempty(y_cut_min)
        y_cut_min = 1;
    end
    if isempty(y_cut_max)
        y_cut_max = img_size(2);
    end
    
    % 去除多余的行和列
    img_p = img(x_cut_min:x_cut_max,y_cut_min:y_cut_max);
    mask_p = mask(x_cut_min:x_cut_max,y_cut_min:y_cut_max); 
    
    % resize到aim_size*aim_size
    img_p = imresize(img_p,[stage1_aim_size stage1_aim_size],'bicubic');
    mask_p = imresize(mask_p,[stage1_aim_size stage1_aim_size],'nearest');
end

