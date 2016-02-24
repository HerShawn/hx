% 2015/12/3
% 2015/12/7 ����/evaluation
% 2015/12/9 DICT&WRA DICT&EDIT_DISTANCE
% 2015/12/10 ��DICT&WRAΪ������߷ָ�Ч����
clc
clear
close all
addpath('D:\hx\piotr_toolbox');
addpath(genpath(pwd));
addpath('F:\Program Files\matlab\toolbox');
% Parameters for EdgeBox
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect
do_dir='D:\hx\edgebox-contour-neumann\';
dir_img = dir([do_dir 'train-textloc\*.jpg'] );
num_img = length(dir_img);
precision = []; recall = []; fscore = [];
global totalTrueBbox totalPredBbox totalGoodBbox total_edit_distance;
totalTrueBbox=0;
totalPredBbox=0;
totalGoodBbox=0;
total_edit_distance=0;
for indexImg =1:num_img
    %% �ֶ�λ�׶�
    disp(['��' num2str(indexImg+99) '��ͼ']);
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    img_name = [do_dir 'train-textloc\' img_value '.jpg'];
    g = imread(img_name);
    %���ƻ���neumann��contour��һ���Ӻ��������ںϣ�����������txt�ļ���
    txt_name = [do_dir 'addContour2Neumann2\' img_value '.txt'];
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
    gt=[txt_data{:,1} txt_data{:,2} txt_data{:,3} txt_data{:,4}];
    
    gt_txt_name=[do_dir 'train-textloc\gt_' img_value '.txt'];
    fid = fopen(gt_txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d,%s');
    fclose(fid);
    num_gt = length(txt_data{2});
    img_gt = cell(num_gt,5);
    for i = 1:num_gt
        img_gt(i,1) = {txt_data{1}(i)};
        img_gt(i,2) = {txt_data{2}(i)};
        img_gt(i,3) = {txt_data{3}(i)};
        img_gt(i,4) = {txt_data{4}(i)};
        img_gt(i,5) = {txt_data{5}{i}(:,2:end-1)};
    end
    % ���neumann%contourû��⵽������edgebox
    % 1/19��size(gt,1)==0||size(gt,1)==1��Ϊsize(gt,1)==0
    if size(gt,1)==0
        gt=coarse_localization(g,gt,model,opts);
    else
    end

%     target_txt_name = [do_dir 'coarse_localization3\coarse_' img_value '.txt'];
%     dlmwrite(target_txt_name, gt);

    %% �ֶ�λ���˽���
    %% ϸ��λ�׶�
    %CHAR&WRA
    %     fine_localization_12_7(img_gt,gt,g);
    %CHAR&EDIT DISTANCE
    %     fine_localization_12_8(img_gt,gt,g);
    % DICT&WRA
%     fine_bboxes=fine_localization_dict_wra(img_gt,gt,g);
   
%     fine_bboxes=fine_bboxes(:,1:4);
%     fine_bboxes(:,3)=fine_bboxes(:,1)+fine_bboxes(:,3);
%     fine_bboxes(:,4)=fine_bboxes(:,2)+fine_bboxes(:,4);

    % DICT&EDIT DISTANCE
    %     fine_localization_dict_ed(img_gt,gt,g);
    
    target_txt_name = [do_dir 'fine_localization\' img_value '.txt'];
    %2016/2/22 ���Ž�������ύ��ICDAR2015���в���
%     target_txt_name = [do_dir 'fine_localization\res_img_' num2str(indexImg) '.txt'];
    dlmwrite(target_txt_name, gt);
    
    figure(indexImg);
%     fine_bboxes=fine_bboxes(:,1:4);
    bbGt('showRes',g,gt,gt);
    save_name=[do_dir 'fine_localization\' img_value '.jpg'];
    print(indexImg, '-dpng', save_name);
    
    %% ϸ��λ���˽���
end
%% �����׶�
%  fine_localization_eval2()
 WRA(precision,recall,fscore);
% fprintf('TOTAL_EDIT_DISTANCE =%d\n', total_edit_distance);
%% �������˽���