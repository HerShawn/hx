
% 2015/11/30
%���汾��Ŀ�ģ�
% �ֶ�λ�Ľ����neumann&contour&edgebox
%�ɲ����²�ȫ��


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



for indexImg = 14:17
    
    %% �ֶ�λ�׶�
    disp(['��' num2str(indexImg+99) '��ͼ']);
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    
    img_name = [do_dir 'train-textloc\' img_value '.jpg'];
    g = imread(img_name);
    [wid,high,~]=size(g);
%     �𼱣����nuemann&contourû�м�⵽������edgebox;
%   ���ߣ���⵽�ˣ���ʶ�����С����ֵ������edgebox����Ҳ��
%   integrated������ʶ��׶���Ϣ���������׶Σ����������Ƿָ
%     tic, bbs=edgeBoxes(g,model,opts); toc
    
%���ƻ���neumann��contour��һ���Ӻ��������ںϣ�����������txt�ļ���
    txt_name = [do_dir 'addContour2Neumann\' img_value '.txt'];
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
%     groundtruth.left=txt_data{:,1};
%     groundtruth.top =txt_data{:,2};
%     groundtruth.right=txt_data{:,3};
%     groundtruth.down=txt_data{:,4};
    
    gt=[txt_data{:,1} txt_data{:,2} txt_data{:,3} txt_data{:,4}];
%     gt = [max(groundtruth.left,1) max(groundtruth.top,1) groundtruth.right-groundtruth.left  groundtruth.down-groundtruth.top ];
%     gt(:,5)=0;

    % ���neumann%contourû��⵽������edgebox
    if size(gt,1)==0
    bbs=edgeBoxes(g,model,opts);
    bbs=bbs([1,2,6,7],:);
    bbs=bbs(:,1:4);
    bbs(:,3)=bbs(:,1)+bbs(:,3);
    bbs(:,4)=bbs(:,2)+bbs(:,4);
    gt=zeros(1,4);
    gt(1,1)=min(bbs(:,1));
    gt(1,2)=min(bbs(:,2));
    gt(1,3)=max(bbs(:,3))-gt(1,1);
    gt(1,4)=max(bbs(:,4))-gt(1,2);
    figure(indexImg);
         bbGt('showRes',g,gt,gt);
         save_name=[img_value '.jpg'];
         print(indexImg, '-dpng', save_name);
         
    elseif size(gt,1)==1
        bbs=edgeBoxes(g,model,opts);
        bbs=bbs([1,2,6,7],:);
        bbs=bbs(:,1:4);
        bbs(:,3)=bbs(:,1)+bbs(:,3);
        bbs(:,4)=bbs(:,2)+bbs(:,4);
        gt_bbs=zeros(1,4);
        gt_bbs(1,1)=min(bbs(:,1));
        gt_bbs(1,2)=min(bbs(:,2));
        gt_bbs(1,3)=max(bbs(:,3));
        gt_bbs(1,4)=max(bbs(:,4));
 
        area_gt = gt(1,3).*gt(1,4);
        area_bbs=gt_bbs(1,3).*gt_bbs(1,4);
       
        
        int_area = rectint(gt_bbs, gt)';
        
        if double(int_area) / double(area_bbs)>0.6
            gt_tmp=gt;
        elseif double(int_area) / double(area_gt)>0.6&&double(area_bbs)/double(wid*high)<=0.5
%             gt=[];
            gt_bbs(1,3)=max(bbs(:,3))-gt_bbs(1,1);
            gt_bbs(1,4)=max(bbs(:,4))-gt_bbs(1,2);
            gt_tmp=gt_bbs;
        end
        
      
        
        
        figure(indexImg);
        bbGt('showRes',g,gt,gt);
        save_name=[img_value '.jpg'];
        print(indexImg, '-dpng', save_name);
        
    else    %��⵽�˾Ͳ���edgebox������ܻ���  
        %�����������ʶ��������㣬����edgebox������integrated˼��
    figure(indexImg);
         bbGt('showRes',g,gt,gt);
         save_name=[img_value '.jpg'];
         print(indexImg, '-dpng', save_name);
    end
    
    target_txt_name = [do_dir 'coarse_localization\' img_value '.txt'];
     dlmwrite(target_txt_name, gt,'-append');
    
end