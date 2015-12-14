clc
clear
addpath('C:\Program Files\MATLAB\R2014a');
addpath(genpath(pwd));
dir_img = dir('D:\hx\数据\icdar2011\train\train-textloc\*.jpg');
num_img = length(dir_img);
for indexImg = 1:num_img
     img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    
    txt_name = ['D:\hx\数据\icdar2011\train\train-textloc\gt_' img_value '.txt'];
    
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d,%s');
    groundtruth.left=txt_data{:,1};
    groundtruth.top =txt_data{:,2};
    groundtruth.right=txt_data{:,3};
    groundtruth.down=txt_data{:,4};
    save_name=['D:\hx\数据\icdar2011\train\train-textloc\gt_' img_value '.mat'];
    save(save_name,'groundtruth');

    fclose(fid);
end
% txtName = 'G:\数据\icdar2011\test-textloc\gt_101.txt';
% fid = fopen(txtName);
% txt_data = textscan(fid,'%d,%d,%d,%d,%s');
% imwrite(img,[save_dir 'detection\' img_value '.mat']);
% fclose(fid);