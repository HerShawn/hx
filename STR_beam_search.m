clc
clear
close all
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

for indexImg = 1:1
    
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
    
    gt = [max(groundtruth.left,1) max(groundtruth.top,1) groundtruth.right-groundtruth.left  groundtruth.down-groundtruth.top ];
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
  
    
    %% 细定位阶段 & 识别模块
    addpath(genpath('../detectorDemo'));
    img_result_gt=[];
    img_result_dt=[];
    run model_release/matconvnet/matlab/vl_setupnn.m
    % 【1】先处理基于er与contour方法得到的gtRes2
    
    %【1.1】如果gtRes2只有一个，那就别判断了，直接识别就好了。
    if size(gtRes2,1)==1
        im=g(max(gtRes2(1,2),1):min((gtRes2(1,2)+gtRes2(1,4)),len),max(gtRes2(1,1),1):min((gtRes2(1,1)+gtRes2(1,3)),wid));
        %             figure;imshow(im);
        if size(im, 3) > 1, im = rgb2gray(im); end;
        im = imresize(im, [32, 100]);
        im = single(im);
        s = std(im(:));
        im = im - mean(im(:));
        im = im / ((s + 0.0001) / 128.0);
        net = load('dictnet.mat');
        lexicon = load_nostruct('lex.mat');
        stime = tic;
        res = vl_simplenn(net, im);
        fprintf('Detection %.2fs\n', toc(stime));
        [score,lexidx] = max(res(end).x(:));
        fprintf(' %s\t%f\n', lexicon{lexidx},score);
        %                     im_tmp1=[];
        %                     im_tmp=[];
        %             net = load('charnet.mat');
        %             stime = tic;
        %             res = vl_simplenn(net, im);
        %             % fprintf('Detection %.2fs\n', toc(stime));
        %             s = '0123456789abcdefghijklmnopqrstuvwxyz ';
        %             [~,pred] = max(res(end).x, [], 1);
        %             fprintf('Predicted text: %s\n', s(pred));
    else 
        %【1.2】如果gtRes2不只有一个，我们才接着进行细定位（CNN检测）
        if size(gtRes2,1)>1
            for i=1:size(gtRes2,1)         
                im=g(max(gtRes2(i,2),1):min((gtRes2(i,2)+gtRes2(i,4)),len),gtRes2(i,1):min((gtRes2(i,1)+gtRes2(i,3)),wid));
                wid_2= size(im,2);
                hig_2=size(im,1);
                img_result_gt=runDetectorDemo(im);
                if size(img_result_gt,1)>0
                    %这里是细定位得到的结果img_result_gt
                    for j=1:size(img_result_gt,1)
                        im_tmp1=im(max(img_result_gt(j,2),1):min((img_result_gt(j,2)+img_result_gt(j,4)),(hig_2-img_result_gt(j,2))),img_result_gt(j,1):min((img_result_gt(j,1)+img_result_gt(j,3)),wid_2));            
                        %（在这里改了下细定位横坐标不足的问题 11月5号）
                        if j==1     %解决尺度导致开头缺失问题； 但是103的那个就不好解决。
                            %                          figure;imshow(im);
                            im_tmp1=im(max(img_result_gt(j,2),1):min((img_result_gt(j,2)+img_result_gt(j,4)),(hig_2-img_result_gt(j,2))),1:(0+img_result_gt(j,3)));
                        end         %若开头就是结尾，那么要解决开头和结尾缺失问题（尺度导致）
                        if j==size(img_result_gt,1)&&size(img_result_gt,1)==1
                            %                         figure;imshow(im);
                            im_tmp1=im(max(img_result_gt(j,2),1):min((img_result_gt(j,2)+img_result_gt(j,4)),(hig_2-img_result_gt(j,2))),1:wid_2);
                        end         %如果分割成几块（开头不是结尾），那么要解决结尾缺失问题。
                        if j==size(img_result_gt,1)&&size(img_result_gt,1)>1
                            %                          figure;imshow(im);
                            im_tmp1=im(max(img_result_gt(j,2),1):min((img_result_gt(j,2)+img_result_gt(j,4)),(hig_2-img_result_gt(j,2))),img_result_gt(j,1):wid_2);
                        end
                        %                 
                        %                      figure;imshow(im_tmp1);
                        if size(im_tmp1, 3) > 1, im_tmp1 = rgb2gray(im_tmp1); end;
                        im_tmp = imresize(im_tmp1, [32, 100]);
                        im_tmp = single(im_tmp);
                        s = std(im_tmp(:));
                        im_tmp = im_tmp - mean(im_tmp(:));
                        im_tmp = im_tmp / ((s + 0.0001) / 128.0);
                        net = load('dictnet.mat');
                        lexicon = load_nostruct('lex.mat');
                        stime = tic;
                        res = vl_simplenn(net, im_tmp);
                        fprintf('Detection %.2fs\n', toc(stime));
                        [score,lexidx] = max(res(end).x(:));
                        fprintf(' %s\t%f\n', lexicon{lexidx},score);
                        im_tmp1=[];
                        im_tmp=[];
                        %             net = load('charnet.mat');
                        %             stime = tic;
                        %             res = vl_simplenn(net, im);
                        %             % fprintf('Detection %.2fs\n', toc(stime));
                        %             s = '0123456789abcdefghijklmnopqrstuvwxyz ';
                        %             [~,pred] = max(res(end).x, [], 1);
                        %             fprintf('Predicted text: %s\n', s(pred));
                    end  
                end
            end  
        end
    end
    %以上是gtRes，也就是基于er与contour的方法（粗定位）所得到Bboxes再经过细定位处理后所得结果；
    
    
    
    %【2】下面是基于edgebox的方法所得的bboxes经细定位处理后所得结果；
    %【2.1】在dtRes中经过细定位处理后是检测到
    if size(dtRes,1)>0
        for i=1:size(dtRes,1)
            im=g(dtRes(i,2):(dtRes(i,2)+dtRes(i,4)),dtRes(i,1):(dtRes(i,1)+dtRes(i,3)));
            wid_3= size(im,2);
            hig_3=size(im,1);
            img_result_dt=runDetectorDemo(im);
            if size(img_result_dt,1)>0
                for j=1:size(img_result_dt,1)
                    im_dt=im(img_result_dt(j,2):min((img_result_dt(j,2)+img_result_dt(j,4)),hig_3-img_result_dt(j,2)),img_result_dt(j,1):min((img_result_dt(j,1)+img_result_dt(j,3)),wid_3));   
                    %（在这里改下细定位横坐标不足的问题 11月5号）
                    if j==1
                        %                          figure;imshow(im);
                        im_dt=im(img_result_dt(j,2):min((img_result_dt(j,2)+img_result_dt(j,4)),(hig_3-img_result_dt(j,2))),1:(0+img_result_dt(j,3)));
                    end
                    if j==size(img_result_dt,1)&&size(img_result_dt,1)==1
                        im_tmp1=im(img_result_dt(j,2):min((img_result_dt(j,2)+img_result_dt(j,4)),(hig_3-img_result_dt(j,2))),1:wid_3);
                    end
                    if j==size(img_result_dt,1)&&size(img_result_dt,1)>1
                        im_tmp1=im(img_result_dt(j,2):min((img_result_dt(j,2)+img_result_dt(j,4)),(hig_3-img_result_dt(j,2))),img_result_dt(j,1):wid_3);
                    end
                    % 
                    %                      figure;imshow(im_dt);
                    if size(im_dt, 3) > 1, im_dt = rgb2gray(im_dt); end;
                    im_dt_2 = imresize(im_dt, [32, 100]);
                    im_dt_2 = single(im_dt_2);
                    s = std(im_dt_2(:));
                    im_dt_2 = im_dt_2 - mean(im_dt_2(:));
                    im_dt_2 = im_dt_2 / ((s + 0.0001) / 128.0);
                    net = load('dictnet.mat');
                    lexicon = load_nostruct('lex.mat');
                    stime = tic;
                    res = vl_simplenn(net, im_dt_2);
                    fprintf('Detection %.2fs\n', toc(stime));
                    [score,lexidx] = max(res(end).x(:));
                    fprintf(' %s\t%f\n', lexicon{lexidx},score);
                    %             net = load('charnet.mat');
                    %             stime = tic;
                    %             res = vl_simplenn(net, im);
                    %             % fprintf('Detection %.2fs\n', toc(stime));
                    %             s = '0123456789abcdefghijklmnopqrstuvwxyz ';
                    %             [~,pred] = max(res(end).x, [], 1);
                    %             fprintf('Predicted text: %s\n', s(pred));
                end
            else
                %如果没有在dt的粗定位中得到细定位结果，那就也直接对粗定位识别！
                %【2.2】11月6号，也许我可以通过识别模块的score来进一步过滤呢~
                if size(im, 3) > 1, im = rgb2gray(im); end;
                im = imresize(im, [32, 100]);
                im = single(im);
                s = std(im(:));
                im = im - mean(im(:));
                im = im / ((s + 0.0001) / 128.0);
                net = load('dictnet.mat');
                lexicon = load_nostruct('lex.mat');
                stime = tic;
                res = vl_simplenn(net, im);
                fprintf('Detection %.2fs\n', toc(stime));
                [score,lexidx] = max(res(end).x(:));
                fprintf(' %s\t%f\n', lexicon{lexidx},score);
                %                 net = load('charnet.mat');
                %                 stime = tic;
                %                 res = vl_simplenn(net, im);
                %                 % fprintf('Detection %.2fs\n', toc(stime));
                %                 s = '0123456789abcdefghijklmnopqrstuvwxyz ';
                %                 [~,pred] = max(res(end).x, [], 1);
                %                 fprintf('Predicted text: %s\n', s(pred));
            end
        end
    end
   
end