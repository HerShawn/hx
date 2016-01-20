clc
clear
close all
addpath('D:\hx\piotr_toolbox');
addpath(genpath(pwd));
addpath('F:\Program Files\matlab\toolbox');

do_dir='D:\hx\edgebox-contour-neumann\';
dir_img = dir([do_dir 'train-textloc\*.jpg'] );
num_img = length(dir_img);

for indexImg =225:num_img
    %% 两粗定位比较
    disp(['第' num2str(indexImg+99) '张图']);
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    img_name = [do_dir 'train-textloc\' img_value '.jpg'];
    g = imread(img_name);
    
    
    txt_name = [do_dir 'coarse_localization\' img_value '.txt'];
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
    gt1=[txt_data{:,1} txt_data{:,2} txt_data{:,3} txt_data{:,4}];
    
    
    txt_name = [do_dir 'coarse_localization2\' img_value '.txt'];
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
    
    
    gt2=[txt_data{:,1} txt_data{:,2} txt_data{:,3} txt_data{:,4} txt_data{:,1}];
    
    
    figure(indexImg);
%      bbGt('showRes',g,gt1);
     bbGt1('showRes',g,gt1,gt2);
    
    save_name=[do_dir 'coarse_localization_diff\' img_value '.jpg'];
    print(indexImg, '-dpng', save_name);
    
 
    
end