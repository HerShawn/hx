% 2015/11/30
%本版本的目的：
% 粗定位的结果：neumann&contour&edgebox
%可测评下查全率
clc
clear
close all
addpath('C:\Users\Text\Desktop\edge box\release\piotr_toolbox');
addpath(genpath(pwd));
addpath('C:\Program Files\MATLAB\R2014a\toolbox');

% Parameters for EdgeBox
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

do_dir='C:\Users\Text\Desktop\edgebox-contour-neumann三种检测方法的比较\';
dir_img = dir([do_dir 'train-textloc\*.jpg'] );
num_img = length(dir_img);
for indexImg =1:7
    %% 粗定位阶段
    disp(['第' num2str(indexImg+99) '张图']);
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    img_name = [do_dir 'train-textloc\' img_value '.jpg'];
    g = imread(img_name);
    
%按计划，neumann和contour在一个子函数里面融合，将结果存入此txt文件中
    txt_name = [do_dir 'addContour2Neumann\' img_value '.txt'];
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
    gt=[txt_data{:,1} txt_data{:,2} txt_data{:,3} txt_data{:,4}];
    % 如果neumann%contour没检测到，才用edgebox
    if size(gt,1)==0
    gt=coarse_localization(g,model,opts);
    figure(indexImg);
         bbGt('showRes',g,gt,gt);
         save_name=[img_value '.jpg'];
         print(indexImg, '-dpng', save_name);           
    else    %检测到了就不用edgebox，否则很混乱  
        %但是如果后续识别分数不足，再用edgebox，这是integrated思想
    figure(indexImg);
         bbGt('showRes',g,gt,gt);
         save_name=[img_value '.jpg'];
         print(indexImg, '-dpng', save_name);
    end
    %% 粗定位到此结束
    
    %% 细定位阶段
    
    
    
    %% 细定位到此结束
    
end