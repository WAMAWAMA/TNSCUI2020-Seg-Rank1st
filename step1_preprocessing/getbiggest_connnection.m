function [mask_img] = getbiggest_connnection(mask_img)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明





%保留最大连通域
disp('Preserve the maximum connected domain')
cc = bwconncomp(mask_img,8);
numPixels = cellfun(@numel,cc.PixelIdxList);
[~,idx] = max(numPixels);
mask_img(cc.PixelIdxList{idx}) = 2;
mask_img = double(mask_img>1.5);

end

