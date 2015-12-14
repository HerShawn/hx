clc
clear
% warning off all

addpath('C:\Users\Administrator\Desktop\edge box\release\piotr_toolbox');
addpath(genpath(pwd));


%% Parameters for EdgeBox 
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%%
do_dir='C:\Users\Text\Desktop\edgebox-contour-neumann三种检测方法的比较\';
dir_img = dir([do_dir 'train-textloc\*.jpg'] );
% save_dir = 'D:\hx\数据\Train_Data\';

num_img = length(dir_img);
% for file_num =1:233
% for indexImg = 166:num_img
 
for indexImg = 1:num_img
    disp(['第' num2str(indexImg+99) '张图']);
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    
%     switch indexImg
%         case 1
%             img_value='126';
%         case 1
%             img_value='127';
%         case 2
%             img_value='147';
%         case 1
%             img_value='148';
%         case 1
%             img_value='320';  
%     end
%     Image_Path =[File_Path,num2str(file_num),'.jpg'];
%     g = imread(Image_Path);
   
%     img_name = ['C:\Users\Text\Desktop\edgebox-contour-neumann三种检测方法的比较\neumann_2011train-detectionc\' img_value '.jpg'];
   img_name = [do_dir 'train-textloc\' img_value '.jpg'];
   g = imread(img_name);
    [len,wid,~] = size(g);
    tic, bbs=edgeBoxes(g,model,opts); toc
%     Gt_Path =['D:\hx\数据\icdar2011\train\train-textloc\gt_' img_value '.mat'];
%     load(Gt_Path);

    txt_name = [do_dir 'addContour2Neumann\' img_value '.txt'];
    
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    groundtruth.left=txt_data{:,1};
    groundtruth.top =txt_data{:,2};
    groundtruth.right=txt_data{:,3};
    groundtruth.down=txt_data{:,4};
     
%     groundtruth.top = max(1,groundtruth.top);
%     groundtruth.left = max(1,groundtruth.left);
%     groundtruth.down = min(len,groundtruth.down);
%     groundtruth.right =min(wid,groundtruth.right);

    gt = [groundtruth.left groundtruth.top groundtruth.right-groundtruth.left+1  groundtruth.down-groundtruth.top+1 ];

    gt(:,5)=0; 
    [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),.1,1);


    figure(indexImg);
     bbGt('showRes',g,gtRes,dtRes(dtRes(:,6)==1,:));
     save_name=[img_value '.jpg'];
     print(indexImg, '-dpng', save_name);  
%     dtRest_final=dtRes(dtRes(:,6)==1,:);
%     for i=1:size(dtRest_final, 1)
%         g(dtRest_final(i,2):dtRest_final(i,2)+4,dtRest_final(i,1):dtRest_final(i,1)+dtRest_final(i,3)) = 255;
%          g(dtRest_final(i,2)+dtRest_final(i,4)-2:dtRest_final(i,2)+dtRest_final(i,4)+2,dtRest_final(i,1):dtRest_final(i,1)+dtRest_final(i,3)) = 255;
%          g(dtRest_final(i,2):dtRest_final(i,2)+dtRest_final(i,4),dtRest_final(i,1):dtRest_final(i,1)+4) = 255;
%          g(dtRest_final(i,2):dtRest_final(i,2)+dtRest_final(i,4),dtRest_final(i,1)+dtRest_final(i,3)-4:dtRest_final(i,1)+dtRest_final(i,3)) = 255;
%           imwrite(g,[save_dir 'edgebox_detection\' img_value '.jpg']);
%     end
%     for i=1:size(gtRes, 1)
%         g(gtRes(i,2):gtRes(i,2)+4,gtRes(i,1):gtRes(i,1)+gtRes(i,3)) = 1;
%          g(gtRes(i,2)+gtRes(i,4)-2:gtRes(i,2)+gtRes(i,4)+2,gtRes(i,1):gtRes(i,1)+gtRes(i,3)) = 1;
%          g(gtRes(i,2):gtRes(i,2)+gtRes(i,4),gtRes(i,1):gtRes(i,1)+4) = 1;
%          g(gtRes(i,2):gtRes(i,2)+gtRes(i,4),gtRes(i,1)+gtRes(i,3)-4:gtRes(i,1)+gtRes(i,3)) = 1;
%          imwrite(g,[save_dir 'edgebox_detection\' img_value '.jpg']);
%     end
    
    
    title('green=matched gt  red=missed gt  dashed-green=matched detect');
end