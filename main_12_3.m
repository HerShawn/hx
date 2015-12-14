% 2015/12/3
% 2015/12/7 ����/evaluation
%���汾��ϸ��λ���������⣬�ָʶ�𣻺���
% �˰汾����ı��зָ�����⣻
% 12/3Ҫ�����ı��еġ����ӻ�����
%���빤����Ҫ��CNN_Dector��visualizeBoxes��
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
do_dir='C:\Users\Text\Desktop\edgebox-contour-neumann���ּ�ⷽ���ıȽ�\';
dir_img = dir([do_dir 'train-textloc\*.jpg'] );
num_img = length(dir_img);
for indexImg =65:65
    %% �ֶ�λ�׶�
    disp(['��' num2str(indexImg+99) '��ͼ']);
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    img_name = [do_dir 'train-textloc\' img_value '.jpg'];
    g = imread(img_name);   
%���ƻ���neumann��contour��һ���Ӻ��������ںϣ�����������txt�ļ���
    txt_name = [do_dir 'addContour2Neumann\' img_value '.txt'];
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
    gt=[txt_data{:,1} txt_data{:,2} txt_data{:,3} txt_data{:,4}];
    % ���neumann%contourû��⵽������edgebox
    if size(gt,1)==0
    gt=coarse_localization(g,model,opts);
%     figure(indexImg);
%          bbGt('showRes',g,gt,gt);
%          save_name=[img_value '.jpg'];
%          print(indexImg, '-dpng', save_name);           
    else    %��⵽�˾Ͳ���edgebox������ܻ���  
        %�����������ʶ��������㣬����edgebox������integrated˼��
%     figure(indexImg);
%          bbGt('showRes',g,gt,gt);
%          save_name=[img_value '.jpg'];
%          print(indexImg, '-dpng', save_name);
    end
    %% �ֶ�λ���˽���
    %% ϸ��λ�׶�
    [wbboxes,predwords]=fine_localization_12_7(gt,g);
    %% ϸ��λ���˽���
    %% �����׶�
    
    %% �������˽���
end