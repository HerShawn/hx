% 12/9 DICT&WRA
%12/16 debug
%1/13 将对整个数据集的测评改为对每张图片的测评，并且挑出拉低分数的进行追踪改进。
function fine_bboxes =fine_localization_dict_wra(img_gt,gtRes2,g)
addpath(genpath('../detectorDemo'));
run model_release/matconvnet/matlab/vl_setupnn.m
global totalTrueBbox totalPredBbox totalGoodBbox;
fine_bboxes=[];
wbboxes = [];
predwords = [];
[height,width,~] = size(g);
%如果粗定位后的结果只有一个bounding box，那么直接求识别分数；若不足阈值，再一遍粗定位（CNN？edgebox?)
% 粗定位只得到一个bbox时，无需beam search/分割；而要做识别和后处理。
if size(gtRes2,1)==1&&gtRes2(1,3)/gtRes2(1,4)<10
    
    %1/20 细定位查全只有47%很大原因在这里没有求fine_bboxes
    
    tempbbox_1=zeros(1,5);
    
    tempbbox_1(2)=gtRes2(1,2);
  
    tempbbox_1(4) =gtRes2(1,4);
   
    tempbbox_1(1) =gtRes2(1,1);
    
    tempbbox_1(3) =gtRes2(1,3);
    
    tempbbox_1(5)= 2;
    
    fine_bboxes=[fine_bboxes;tempbbox_1];
    
    im=g(max(gtRes2(1,2),1):min((gtRes2(1,2)+gtRes2(1,4)),height),max(gtRes2(1,1),1):min((gtRes2(1,1)+gtRes2(1,3)),width));
    if size(im, 3) > 1, im = rgb2gray(im); end;
    im = imresize(im, [32, 100]);
    im = single(im);
    s = std(im(:));
    im = im - mean(im(:));
    im = im / ((s + 0.0001) / 128.0);
    net = load('dictnet.mat');
    lexicon = load_nostruct('lex.mat');
%     stime = tic;
    res = vl_simplenn(net, im);
%     fprintf('DICT Detection %.2fs\n', toc(stime));
    [score,lexidx] = max(res(end).x(:));
    fprintf(' %s\t%f\n', lexicon{lexidx},score);
    %两者的net是不一样的！！
    totalPredBbox = totalPredBbox+1;
    pred_now=1
    good_ori=totalGoodBbox;
    pred_tag = lexicon{lexidx};
    for tt = 1:size(img_gt,1)
        if  strcmpi(strtrim(pred_tag), cell2mat(img_gt(tt,5)))
            totalGoodBbox = totalGoodBbox+1;
        end
    end
    good_now=totalGoodBbox-good_ori
else
    %细定位CNN检测子
    if size(gtRes2,1)~=0
        pred_ori=totalPredBbox;
        good_ori=totalGoodBbox;
        %对于粗定位得到的每一个文本行：
        for i=1:size(gtRes2,1)
            im=g(max(gtRes2(i,2),1):min((gtRes2(i,2)+gtRes2(i,4)),height),gtRes2(i,1):min((gtRes2(i,1)+gtRes2(i,3)),width));
            global scoreTable wordsTable;
            thresh=0.2;
            response=CNN_Detector(im);
            bboxes = response.bbox;
            spaces = response.spaces;
            hx=[response.chars.locations];
            charnums=size(hx,2);
            c_split=(charnums+18.0)/25.0;
            numbbox = size(bboxes,1);
            % 从粗定位文本行中用CNN检测子进行细定位：
            for bidx = 1:numbbox
                %细定位的过滤步骤：
%                 1.20将thresh由0.7降，提高查全
%               做个试验：不顾及查准（即不考虑bboxes(bidx,5)>0.3）的情况下的测评成绩。bboxes(bidx,5)>0.3 &&
                if  length(spaces(bidx).locations)<5
                    %  beam search/分割从这里就开始了
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
                    locations = spaces(bidx).locations;
                    spacescores = spaces(bidx).scores;
                    locations = locations(spacescores>0.7);
                    spacescores = spacescores(spacescores>0.7);
                    [orig_sorted_locations, sortidx]= sort(locations(:),'ascend');
                    spacescores = spacescores(sortidx);
                    % bug:如果一个候选问本行中检测到两个bboxes，那么不该用stdwidth来来结尾！！
                    [stdheight, stdwidth] = size(im);
                    segs = [  [1; orig_sorted_locations] [orig_sorted_locations; stdwidth]];
                    %                     
                    std_starts = [1; orig_sorted_locations];
                    std_ends = [orig_sorted_locations; stdwidth];
                    numbeams = 10;
                    states = [];
                    numsegs = size(segs,1);
                    scoreTable = ones(numsegs+1, numsegs+1)*(-99);
                    wordsTable = cell(numsegs+1, numsegs+1);
                    curr = 1;
                    while isempty(states) || curr<=size(segs,1)
                        %  这里只是针对CHAR这一种模型的识别，包含在beam search；
                        [newstates ,curr]= beam_search_dict(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                        states = newstates;
                    end
                    %fprintf('CHAR prediction: ')
                    %states{1}
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
                    for ss = 1:length(currwords)
                        tempbbox = zeros(1,5);
                        tempbbox_1=zeros(1,5);
                        tempbbox(2) = aa(1);
                        tempbbox_1(2)=gtRes2(i,2);
                        tempbbox(4) = stdheight;
                        tempbbox_1(4) = tempbbox(4)-1;
                        tempbbox(1) = realstdsegs(ss,1)+aa(2)-1;
                        tempbbox_1(1) =gtRes2(i,1)+realstdsegs(ss,1);
                        tempbbox(3) = realstdsegs(ss,2)-realstdsegs(ss,1)+1;
                        tempbbox_1(3) =tempbbox(3)-1;
                        tempbbox(5) = predscores(ss);
                        tempbbox_1(5)= tempbbox(5);
                        wbboxes = [wbboxes;tempbbox];
                        fine_bboxes=[fine_bboxes;tempbbox_1];
                        predwords{end+1} = currwords{ss};
                    end
                    %if bboxes(bidx,5)>0.8 && length(spaces(bidx).locations)<5
                end   %最内层
                % for bidx = 1:numbbox
            end %一个粗定位文本行内，检测到几个细定位bboxes，然后处理每一个bbox
            % for i=1:size(gtRes2,1)
            if ~exist('pre_numbbox')
            pre_numbbox=0;
            end
           
            pre_numbbox_temp=post_precess(im,numbbox,wbboxes,predwords,thresh,img_gt,pre_numbbox);
            
            pre_numbbox=pre_numbbox_temp;
        end %看有几个粗定位文本行，然后处理每一个粗定位文本行
        % if size(gtRes2,1)>1
        pred_now=totalPredBbox-pred_ori
        good_now=totalGoodBbox-good_ori
    end %粗定位检测到多个文本行的情况
    % if size(gtRes2,1)==1
end %粗定位只检测到一个文本行的情况
totalTrueBbox = totalTrueBbox+size(img_gt,1)
truth_now=size(img_gt,1)
if pred_now==0
    img_p=0
else
    img_p=good_now/pred_now
end
img_r=good_now/truth_now
if img_p==0&&img_r==0
    img_f=0
else
    img_f =  2*img_p*img_r/(img_p+img_r)
end

end