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

precision = []; recall = []; fscore = [];
totalTrueBbox = 0;
totalPredBbox = 0;
totalGoodBbox = 0;

for indexImg = 13:13
    
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
    fclose(fid);
    groundtruth.left=txt_data{:,1};
    groundtruth.top =txt_data{:,2};
    groundtruth.right=txt_data{:,3};
    groundtruth.down=txt_data{:,4};
    
    
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
    
    %     img_gt.left=txt_data{:,1};
    %     img_gt.top =txt_data{:,2};
    %     img_gt.right=txt_data{:,3};
    %     img_gt.down=txt_data{:,4};
    %     img_gt.label=lower(char(txt_data{:,5}));
    
    
    %     groundtruth.top = max(1,groundtruth.top);
    %     groundtruth.left = max(1,groundtruth.left);
    %     groundtruth.down = min(len,groundtruth.down);
    %     groundtruth.right =min(wid,groundtruth.right);
    
    gt = [max(groundtruth.left,1) max(groundtruth.top,1) groundtruth.right-groundtruth.left  groundtruth.down-groundtruth.top ];
    gt(:,5)=0;
    [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),1);
    num_gt = length(txt_data{2});
    %     img_gt=[max(img_gt.left,1) max(img_gt.top,1) img_gt.right-img_gt.left  img_gt.down-img_gt.top ];
    
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
    wbboxes = []; % predicted word bboxes
    predwords = []; % predicted labels
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
        totalPredBbox = totalPredBbox+1
        pred_tag = lexicon{lexidx}
        
        % iterate over ground truth bboxes
        for tt = 1:size(img_gt,1)
            % both ground truth and predicted boxes havent been taken, and string matches
            if  strcmp(pred_tag, lower(cell2mat(img_gt(tt,5))))
                
                totalGoodBbox = totalGoodBbox+1
                % predicted and ground truth bboxes are
                % considered 'taken', so that we don't double count
            end
        end
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
                numbbox1 = size(bboxes,1);
                [height, width, ~] = size(im);
                for bidx = 1:numbbox1
                    if bboxes(bidx,5)>0.7 && length(spaces(bidx).locations)<5
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
                        %                         states = [];
                        numsegs = size(segs,1);
                        %                         scoreTable = ones(numsegs+1, numsegs+1)*(-99);
                        %                         wordsTable = cell(numsegs+1, numsegs+1);
                        %                         curr = 1;
                        %                         while isempty(states) || curr<=size(segs,1)
                        %                             [newstates ,curr]= beam_search(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                        %                             states = newstates;
                        %                         end
                        %                         fprintf('CHAR prediction: ')
                        %                         states{1}
                        
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
                        
                        if length(states{1}.path)==1 && states{1}.path(1)==2
                            states{1}.path(1) = 5;
                        end
                        
                        % now generate word bboxes from beam search results
                        startings = states{1}.path==1 | states{1}.path == 2;
                        endings  = states{1}.path==1 | states{1}.path == 3;
                        assert(sum(startings) == sum(endings));
                        realstdsegs = [std_starts(startings) std_ends(endings)];
                        
                        currwords = [];%predicted words in the current line
                        for ww = 1:length(states{1}.words)
                            if ~isempty(states{1}.words{ww})
                                currwords{end+1} = states{1}.words{ww};
                            end
                        end
                        
                        predscores = states{1}.scores(states{1}.scores>thresh);
                        tempbbox = zeros(1,5);
                        for ss = 1:length(currwords)
                            tempbbox = zeros(1,5);
                            tempbbox(2) = aa(1);
                            tempbbox(4) = stdheight;
                            %                              subscores = origscores(:, realstdsegs(ss,1):realstdsegs(ss,2));
                            %                              % compute actual left and right bounds for the current segment
                            %                              [~, ~, ~, bounds] =  score2WordBounds(subscores, currwords(ss));
                            tempbbox(1) = realstdsegs(ss,1)+aa(2)-1;
                            %                              +round((bounds(1)-1)/32*subheight); % adjust x position
                            tempbbox(3) = realstdsegs(ss,2)-realstdsegs(ss,1)+1;
                            %                              - round((bounds(1)+bounds(2)-1)/32*subheight); % adjust width
                            tempbbox(5) = predscores(ss);
                            wbboxes = [wbboxes;tempbbox];
                            predwords{end+1} = currwords{ss};
                        end
                    end
                end
                
                if numbbox1~=0
                    if ~isempty(wbboxes)
                        %remove wbboxes with low recognition scores
                        bad_idx = wbboxes(:,end)<thresh;
                        wbboxes(bad_idx, :) = [];
                        predwords(bad_idx) = [];
                        
                        %sort wbboxes in recognition scores
                        %                     matchScores = wbboxes(:,end);
                        %                     [~, score_idx] = sort(matchScores, 'descend');
                        %                     wbboxes = wbboxes(score_idx,:);
                        %                     predwords = predwords(score_idx);
                        %
                        numbbox2 = size(wbboxes,1);
                        pred_taken = zeros(numbbox2,1);
                        
                    end
                    
                    numbbox2 = size(wbboxes,1);
                    
                    for bidx = numbbox2:numbbox2
                        if pred_taken(bidx)==0
                            x = wbboxes(bidx,1);
                            y = wbboxes(bidx,2);
                            w = wbboxes(bidx,3);
                            h = wbboxes(bidx,4);
                            aa = [max(y,1),max(x,1)];% upper left corner
                            bb = [max(y,1), min(x+w, width)];% upper right corner
                            cc = [min(y+h, height), max(x,1)];% lower left corner
                            fprintf('predword  %s, Recog Score  %2.3f\n', predwords{bidx}, wbboxes(bidx,end));
                            
                            for worse_idx = (bidx+1):numbbox2 % wbboxes that are worse than the current one
                                if pred_taken(worse_idx)==0
                                    x2 = wbboxes(worse_idx,1);
                                    y2 = wbboxes(worse_idx,2);
                                    w2 = wbboxes(worse_idx,3);
                                    h2 = wbboxes(worse_idx,4);
                                    aa2 = [max(y2,1),max(x2,1)]; % upper left corner
                                    bb2 = [max(y2,1), min(x2+w2, width)];% upper right
                                    cc2 = [min(y2+h2, height), max(x2,1)];% lower left
                                    pred_y1 = aa(1); pred_y2 = cc(1);
                                    pred_x1 = aa(2); pred_x2 = bb(2);
                                    pred_rec = [pred_x1, pred_y1, pred_x2-pred_x1+1, pred_y2-pred_y1+1];
                                    pred2_y1 = aa2(1); pred2_y2 = cc2(1);
                                    pred2_x1 = aa2(2); pred2_x2 = bb2(2);
                                    pred2_rec = [pred2_x1, pred2_y1, pred2_x2-pred2_x1+1, pred2_y2-pred2_y1+1];
                                    intersect_area = rectint(pred_rec,pred2_rec);
                                    pred_area = pred_rec(3)* pred_rec(4);
                                    pred2_area = pred2_rec(3)* pred2_rec(4);
                                    if intersect_area>0.5*pred_area || intersect_area>0.5*pred2_area
                                        pred_taken(worse_idx) = 1; % worse bbox did not survive NMS
                                    end
                                end
                            end
                            
                            % make a prediction and evaluate the current bounding box
                            %indicator of whether truebboxes are taken
                            taken = zeros(length(img_gt),1);
                            totalPredBbox = totalPredBbox+1
                            pred_tag = predwords{bidx}
                            
                            % iterate over ground truth bboxes
                            for tt = 1:size(img_gt,1)
                                % both ground truth and predicted boxes havent been taken, and string matches
                                if taken(tt)==0 && pred_taken(bidx)==0 && strcmp(pred_tag, lower(cell2mat(img_gt(tt,5))))
                                    
                                    totalGoodBbox = totalGoodBbox+1
                                    % predicted and ground truth bboxes are
                                    % considered 'taken', so that we don't double count
                                    pred_taken(bidx)=1;
                                    taken(tt) = 1;
                                    
                                end
                            end
                        end
                    end
                end
                %                 wbboxes=[];
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
            
            totalPredBbox = totalPredBbox+1
            pred_tag = lexicon{lexidx}
            
            % iterate over ground truth bboxes
            for tt = 1:size(img_gt,1)
                % both ground truth and predicted boxes havent been taken, and string matches
                if  strcmp(pred_tag, lower(cell2mat(img_gt(tt,5))))
                    
                    totalGoodBbox = totalGoodBbox+1
                    % predicted and ground truth bboxes are
                    % considered 'taken', so that we don't double count
                end
            end
        end
    end
    totalTrueBbox = totalTrueBbox+size(img_gt,1)
end

precision = [precision totalGoodBbox/totalPredBbox]
recall = [recall totalGoodBbox/totalTrueBbox]
if precision==0&&recall==0
    fscore=0
else
    fscore = [fscore 2*precision(end)*recall(end)/(precision(end)+recall(end))]
end