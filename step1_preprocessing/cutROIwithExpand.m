function [img_p,mask_p] = cutROIwithExpand(img,mask,aim_size)
    size_mask = size(mask);
    
    % 找mask最小外界矩阵坐标,长宽，中心点
    index = find(mask);
    [index_dim1,index_dim2] = ind2sub(size(mask),index);
    dim1_len = -(min(index_dim1)-max(index_dim1));
    dim2_len = -(min(index_dim2)-max(index_dim2));
    max_len = max(dim1_len,dim2_len);
    dim1_center = round(0.5*(min(index_dim1)+max(index_dim1)));
    dim2_center = round(0.5*(min(index_dim2)+max(index_dim2)));
    
    % 以下标准是基于256尺寸输入图片的标准,thre为80，外扩分别为20和30
    if max_len > round((size_mask(1)/256) * 80)
        final_add = round(max_len/2) + round((size_mask(1)/256) * 20);
    else
        final_add = round(max_len/2) + round((size_mask(1)/256) * 30);
    end
    
    % 计算截取坐标，并判断是否越界
    dim1_cut_min = dim1_center - final_add;
    dim1_cut_max = dim1_center + final_add;
    dim2_cut_min = dim2_center - final_add;
    dim2_cut_max = dim2_center + final_add;
    
    if dim1_cut_min<1
        dim1_cut_min =1;
    end
    if dim2_cut_min<1
        dim2_cut_min =1;
    end
    if dim1_cut_max>size_mask(1)
        dim1_cut_max =size_mask(1);
    end
    if dim2_cut_max>size_mask(1)
        dim2_cut_max =size_mask(1);
    end
    
    % 截取并保存
    mask = mask(dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max);
    img = img(dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max);
    
    img_p = imresize(img,[aim_size aim_size],'bicubic');
    mask_p = imresize(mask,[aim_size aim_size],'nearest');
end

