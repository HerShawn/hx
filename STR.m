clc
clear
% warning off all

addpath('C:\Users\Text\Desktop\edge box\release\piotr_toolbox');
addpath(genpath(pwd));


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

for indexImg = 52:52
    
    %% 粗定位阶段
    disp(['第' num2str(indexImg+99) '张图']);
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4);
    
    img_name = [do_dir 'train-textloc\' img_value '.jpg'];
    g = imread(img_name);
    [len,wid,~] = size(g);
    tic, bbs=edgeBoxes(g,model,opts); toc
    
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
    [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),1);
    
    
    % gt的清洗
    area = gtRes(:,3).*gtRes(:,4);
    [area, perm1] = sort(area, 'descend');
    gtRes1 = gtRes(perm1, 1:5);
    gtRes1(:,5)=0;
    ratio=0.3;
    suppressed = ones(size(gtRes1, 1), 1);
    perm2=ones(size(perm1));
    for i=1:size(perm1)
        perm2(i)=i;
    end
    for i=1:(size(gtRes1,1)-1)
        for j=(i+1):size(gtRes1,1)
            if ~suppressed(i)
                continue;
            end
            int = rectint(gtRes1(i,:), gtRes1(j,:))';
            %如果neumann与contour并无相交，则不执行NMS
            if length(int) == 0
                break;
            end
            ratios = double(int) ./ double(max(area(i), area(j)));
            indices = find(ratios > ratio) ;
            suppressed(indices+i-1) = 0;
        end
    end
    gtRes2=gtRes1(perm2(suppressed == 1),1:5);
    %gt的清洗完成。

    %dt的清洗
    flag=0;
    dtRes=dtRes([1,2,6,7],:);
    dtRes(:,6)=0;
    not_int_num=0;
    %     dtRes(:,5)=0;
    if size(gtRes2,1)>1
        for i=1:size(dtRes,1)
            for j=1:size(gtRes2,1)
                int2 = rectint(dtRes(i,:), gtRes2(j,:))';
                if int2==0
                    not_int_num=not_int_num+1;
                end
            end
            if not_int_num==size(gtRes2,1)
                dtRes(i,6)=1;
            end
            not_int_num=0;
        end
    elseif  size(gtRes2,1)==0
        area2 = dtRes(:,3).*dtRes(:,4);
        [area2, perm3] = sort(area2, 'descend');
        dtRes = dtRes(perm3, 1:6);
        if area2(1,:)>(double(len*wid*2)/double(3))
        else
            dtRes(:,3)=dtRes(:,3)+dtRes(:,1);
            dtRes(:,4)=dtRes(:,4)+dtRes(:,2);
            dtRes(1,1:2)=min(dtRes(1:end,1:2));
            dtRes(1,3:4)=max(dtRes(1:end,3:4));
            dtRes(1,6)=1;
            flag=1;
        end    
    else
        area2 = dtRes(:,3).*dtRes(:,4);
        [area2, perm3] = sort(area2, 'descend');
        dtRes = dtRes(perm3, 1:6);
        if area2(1,:)>(double(len*wid)/double(2))
        else
            dtRes(:,3)=dtRes(:,3)+dtRes(:,1);
            dtRes(:,4)=dtRes(:,4)+dtRes(:,2);
            dtRes(1,1:2)=min(dtRes(1:end,1:2));
            dtRes(1,3:4)=max(dtRes(1:end,3:4));
            dtRes(1,6)=1;
            flag=1;
        end     
    end
    dtRes=dtRes(find(dtRes(:,6)==1),:);
    if(flag==1)
        dtRes(:,3)=dtRes(:,3)-dtRes(:,1);
        dtRes(:,4)=dtRes(:,4)-dtRes(:,2);
    end
    %dt的清洗完成
    
%     将结果保存成图片，以便观察实验结果
        figure(indexImg);
         bbGt('showRes',g,gtRes2,dtRes);
         save_name=[img_value '.jpg'];
         print(indexImg, '-dpng', save_name);
    
    
    
    %% 细定位阶段
    
    addpath(genpath('../detectorDemo'));
    
        if size(gtRes2,1)>0
        for i=1:size(gtRes2,1)
            im=g(max(gtRes2(i,2),1):min((gtRes2(i,2)+gtRes2(i,4)),len),max(gtRes2(i,1),1):min((gtRes2(i,1)+gtRes2(i,3)),wid));
                runDetectorDemo(im);
        end
       end

    if size(dtRes,1)>0
        for i=1:size(dtRes,1)
            im=g(dtRes(i,2):(dtRes(i,2)+dtRes(i,4)),dtRes(i,1):(dtRes(i,1)+dtRes(i,3)));
                runDetectorDemo(im);
        end
    end
    
    %% 识别阶段
%     run model_release/matconvnet/matlab/vl_setupnn.m
% 
%     if size(gtRes2,1)>0
%         for i=1:size(gtRes2,1)
%             im=g(max(gtRes2(i,2),1):min((gtRes2(i,2)+gtRes2(i,4)),len),max(gtRes2(i,1),1):min((gtRes2(i,1)+gtRes2(i,3)),wid));
%             if size(im, 3) > 1, im = rgb2gray(im); end;
%             im = imresize(im, [32, 100]);
%             im = single(im);
%             s = std(im(:));
%             im = im - mean(im(:));
%             im = im / ((s + 0.0001) / 128.0);
%             % net = load('dictnet.mat');
%             % lexicon = load_nostruct('lex.mat');
%             % stime = tic;
%             % res = vl_simplenn(net, im);
%             % % fprintf('Detection %.2fs\n', toc(stime));
%             % [~,lexidx] = max(res(end).x(:));
%             % fprintf('Predicted text: %s\n', lexicon{lexidx});
%             net = load('charnet.mat');
%             stime = tic;
%             res = vl_simplenn(net, im);
%             % fprintf('Detection %.2fs\n', toc(stime));
%             s = '0123456789abcdefghijklmnopqrstuvwxyz ';
%             [~,pred] = max(res(end).x, [], 1);
%             fprintf('Predicted text: %s\n', s(pred));
%         end
%     end
%     if size(dtRes,1)>0
%         for i=1:size(dtRes,1)
%             im=g(dtRes(i,2):(dtRes(i,2)+dtRes(i,4)),dtRes(i,1):(dtRes(i,1)+dtRes(i,3)));
%             if size(im, 3) > 1, im = rgb2gray(im); end;
%             im = imresize(im, [32, 100]); 
%             im = single(im);
%             s = std(im(:));
%             im = im - mean(im(:));
%             im = im / ((s + 0.0001) / 128.0);
%             % net = load('dictnet.mat');
%             % lexicon = load_nostruct('lex.mat');
%             % stime = tic;
%             % res = vl_simplenn(net, im);
%             % fprintf('Detection %.2fs\n', toc(stime));
%             % [~,lexidx] = max(res(end).x(:));
%             % fprintf('Predicted text: %s\n', lexicon{lexidx});
%             net = load('charnet.mat');
%             stime = tic;
%             res = vl_simplenn(net, im);
%             % fprintf('Detection %.2fs\n', toc(stime));
%             s = '0123456789abcdefghijklmnopqrstuvwxyz ';
%             [~,pred] = max(res(end).x, [], 1);
%             fprintf('Predicted text: %s\n', s(pred)); 
%         end
%     end
    
    title('green=matched gt  red=missed gt  dashed-green=matched detect');
end