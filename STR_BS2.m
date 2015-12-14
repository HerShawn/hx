clc
clear
close all
% warning off all

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

for indexImg = 25:90
    
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
        fprintf('DICT Detection %.2fs\n', toc(stime));
        [score,lexidx] = max(res(end).x(:));
        fprintf(' %s\t%f\n', lexicon{lexidx},score);
        %两者的net是不一样的！！
        net = load('charnet.mat');
        stime = tic;
        res = vl_simplenn(net, im);
        fprintf('CHAR Detection %.2fs\n', toc(stime));
        s = '0123456789abcdefghijklmnopqrstuvwxyz ';
        [score,~] = max(res(end).x(:));
        [~,pred] = max(res(end).x, [], 1);
        fprintf('Predicted text: %s\t%f\n', s(pred),score);
    else
        %【1.2】如果gtRes2不只有一个，我们才接着进行细定位（CNN检测）
        if size(gtRes2,1)>1
            for i=1:size(gtRes2,1)
                im=g(max(gtRes2(i,2),1):min((gtRes2(i,2)+gtRes2(i,4)),len),gtRes2(i,1):min((gtRes2(i,1)+gtRes2(i,3)),wid));
                global scoreTable wordsTable;
%                 [hight_penalty, wid_penalty, ~] = size(im);
%                 aspectratio=single(wid_penalty)/hight_penalty;
%                 c_split=aspectratio/2;
                thresh=0.2;
                response=CNN_Detector(im);
                %在这里，DICT和CHAR所得到的response是一样的
                bboxes = response.bbox;
                spaces = response.spaces;
                %11月9号修改 size(response.chars,2)问题
                hx=[response.chars.locations];
                charnums=size(hx,2);
%                 c_split=aspectratio/2;
                c_split=(charnums+18.0)/25.0;
%                 c_split=1.3;
                numbbox = size(bboxes,1);
                [height, width, ~] = size(im);
                for bidx = 1:numbbox
                    if bboxes(bidx,5)>0.8 && length(spaces(bidx).locations)<5
                        x = bboxes(bidx,1);
                        y = bboxes(bidx,2);
                        w = bboxes(bidx,3);
                        h = bboxes(bidx,4);
                        % four corners of the bounding box
                        % aa---------------bb
                        %  |                |
                        %  |                |
                        %  |                |
                        %  cc--------------dd
                        aa = [max(y,1),max(x,1)];
                        bb = [max(y,1), min(x+w, width)];
                        cc = [min(y+h, height), max(x,1)];
                        dd = [min(y+h, height), min(x+w, width)];
                        % candidate spaces
                        locations = spaces(bidx).locations;
                        spacescores = spaces(bidx).scores;
                        locations = locations(spacescores>0.7);
                        spacescores = spacescores(spacescores>0.7);
                        [orig_sorted_locations, sortidx]= sort(locations(:),'ascend');
                        spacescores = spacescores(sortidx);
                        [stdheight, stdwidth] = size(im);
                        segs = [  [1; orig_sorted_locations] [orig_sorted_locations; stdwidth]];
                        std_starts = [1; orig_sorted_locations];
                        std_ends = [orig_sorted_locations; stdwidth];
                        numbeams = 60;
                        
                        %！！这里是CHAR识别模式的结果
                        states = [];
                        numsegs = size(segs,1);
                        scoreTable = ones(numsegs+1, numsegs+1)*(-99);
                        wordsTable = cell(numsegs+1, numsegs+1);
                        curr = 1;
                        while isempty(states) || curr<=size(segs,1)
                            [newstates ,curr]= beam_search(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                            states = newstates;
                        end
                        fprintf('CHAR prediction: ')
                        states{1}
                        
                        %！！这里是DICT识别模式的结果
                        states = [];
                        scoreTable = ones(numsegs+1, numsegs+1)*(-99);
                        wordsTable = cell(numsegs+1, numsegs+1);
                        curr = 1;
                         while isempty(states) || curr<=size(segs,1)
                            [newstates ,curr]= beam_search_dict(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                            states = newstates;
                         end
                        fprintf('DICT prediction: ')
                        states{1}
                    end
                end
            end
        end
    end
    %以上是gtRes，也就是基于er与contour的方法（粗定位）所得到Bboxes再经过细定位处理后所得结果；
    
    
    
    %【2】下面是基于edgebox的方法所得的bboxes经细定位处理后所得结果；
    %【2.1】在dtRes中经过细定位处理后是检测到
    if size(dtRes,1)>1
        for i=1:size(dtRes,1)
            im=g(dtRes(i,2):(dtRes(i,2)+dtRes(i,4)),dtRes(i,1):(dtRes(i,1)+dtRes(i,3)));
            %             wid_3= size(im,2);
            %             hig_3=size(im,1);
            %             img_result_dt=runDetectorDemo(im);
            %                 global scoreTable1 wordsTable1;
            [hight_penalty, wid_penalty, ~] = size(im);
            aspectratio=single(wid_penalty)/hight_penalty;
%             c_split=aspectratio/2;
            thresh=0.2;
            response=CNN_Detector(im);
            bboxes = response.bbox;
            spaces = response.spaces;
            hx=[response.chars.locations];
                charnums=size(hx,2);
%             c_split=aspectratio/2;
            c_split=(charnums+18.0)/25.0;
%             c_split=1.3;
            numbbox = size(bboxes,1);
            [height, width, ~] = size(im);
            for bidx = 1:numbbox
                if bboxes(bidx,5)>0.7 && length(spaces(bidx).locations)<15
                    x = bboxes(bidx,1);
                    y = bboxes(bidx,2);
                    w = bboxes(bidx,3);
                    h = bboxes(bidx,4);
                    % four corners of the bounding box
                    % aa---------------bb
                    %  |                |
                    %  |                |
                    %  |                |
                    %  cc--------------dd
                    aa = [max(y,1),max(x,1)];
                    bb = [max(y,1), min(x+w, width)];
                    cc = [min(y+h, height), max(x,1)];
                    dd = [min(y+h, height), min(x+w, width)];
                    % candidate spaces
                    locations = spaces(bidx).locations;
                    spacescores = spaces(bidx).scores;
                    locations = locations(spacescores>0.7);
                    spacescores = spacescores(spacescores>0.7);
                    [orig_sorted_locations, sortidx]= sort(locations(:),'ascend');
                    spacescores = spacescores(sortidx);
                    [stdheight, stdwidth] = size(im);
                    segs = [  [1; orig_sorted_locations] [orig_sorted_locations; stdwidth]];
                    std_starts = [1; orig_sorted_locations];
                    std_ends = [orig_sorted_locations; stdwidth];
                    numbeams = 60;
                    states = [];
                    numsegs = size(segs,1);
                    scoreTable = ones(numsegs+1, numsegs+1)*(-99);
                    wordsTable = cell(numsegs+1, numsegs+1);
                    curr = 1;
                    while isempty(states) || curr<=size(segs,1)
                        [newstates ,curr]= beam_search(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                        states = newstates;
                    end
                    fprintf('CHAR prediction: ')
                    states{1}
                    
                    states = [];
                        scoreTable = ones(numsegs+1, numsegs+1)*(-99);
                        wordsTable = cell(numsegs+1, numsegs+1);
                        curr = 1;
                         while isempty(states) || curr<=size(segs,1)
                            [newstates ,curr]= beam_search_dict(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                            states = newstates;
                         end
                        fprintf('DICT prediction: ')
                        states{1}
                end
            end
        end
    else
        if size(dtRes,1)==1
              im=g(max(dtRes(1,2),1):min((dtRes(1,2)+dtRes(1,4)),len),max(dtRes(1,1),1):min((dtRes(1,1)+dtRes(1,3)),wid));
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
        fprintf('DICT Detection %.2fs\n', toc(stime));
        [score,lexidx] = max(res(end).x(:));
        fprintf(' %s\t%f\n', lexicon{lexidx},score);
        net = load('charnet.mat');
        stime = tic;
        res = vl_simplenn(net, im);
        fprintf('CHAR Detection %.2fs\n', toc(stime));
        s = '0123456789abcdefghijklmnopqrstuvwxyz ';
        [~,pred] = max(res(end).x, [], 1);
        [score,~] = max(res(end).x(:));
        fprintf('Predicted text: %s\t%f\n', s(pred),score);
        end
    end 
end