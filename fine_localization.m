
% 12.1 引入细定位
%一步一步来:先搞定CHAR型识别模型
%为使程序井井有条，将STR_BS2中有关细定位的部分移植到这里
% 细定位分为：“分割/beam search" ”识别“ ”后处理“
%beam search的关键在于调整分割参数，提高分割效果；
%识别的关键在于 结合CHAR、DICT模型，提高识别效果；
%后处理关键在于将识别结果格式化、去冗余，提高测评效果；

function fine_localization(gtRes2,g)
addpath(genpath('../detectorDemo'));
run model_release/matconvnet/matlab/vl_setupnn.m
[height,width,~] = size(g);
%如果粗定位后的结果只有一个bounding box，那么直接求识别分数；若不足阈值，再一遍粗定位（CNN？edgebox?)
% 粗定位只得到一个bbox时，无需beam search/分割；而要做识别和后处理。
if size(gtRes2,1)==1
    im=g(max(gtRes2(1,2),1):min((gtRes2(1,2)+gtRes2(1,4)),height),max(gtRes2(1,1),1):min((gtRes2(1,1)+gtRes2(1,3)),width));
    if size(im, 3) > 1, im = rgb2gray(im); end;
    im = imresize(im, [32, 100]);
    im = single(im);
    s = std(im(:));
    im = im - mean(im(:));
    im = im / ((s + 0.0001) / 128.0);
    net = load('charnet.mat');
    stime = tic;
    res = vl_simplenn(net, im);
    fprintf('CHAR Detection %.2fs\n', toc(stime));
    s = '0123456789abcdefghijklmnopqrstuvwxyz ';
    [score,~] = max(res(end).x(:));
    [~,pred] = max(res(end).x, [], 1);
    fprintf('Predicted text: %s\t%f\n', s(pred),score);
else
%细定位CNN检测子
    if size(gtRes2,1)>1
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
                if bboxes(bidx,5)>0.8 && length(spaces(bidx).locations)<5   
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
                    [stdheight, stdwidth] = size(im);
                    segs = [  [1; orig_sorted_locations] [orig_sorted_locations; stdwidth]];
                    std_starts = [1; orig_sorted_locations];
                    std_ends = [orig_sorted_locations; stdwidth];
                    numbeams = 30;
                    states = [];
                    numsegs = size(segs,1);
                    scoreTable = ones(numsegs+1, numsegs+1)*(-99);
                    wordsTable = cell(numsegs+1, numsegs+1);
                    curr = 1;
                    while isempty(states) || curr<=size(segs,1)
                    %  这里只是针对CHAR这一种模型的识别，包含在beam search；                       
                        [newstates ,curr]= beam_search(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                        states = newstates;
                    end
                    fprintf('CHAR prediction: ')
                    states{1}
                    
                %if bboxes(bidx,5)>0.8 && length(spaces(bidx).locations)<5                     
                end   %最内层
            % for bidx = 1:numbbox               
            end %一个粗定位文本行内，检测到几个细定位bboxes，然后处理每一个bbox
        % for i=1:size(gtRes2,1)            
        end %看有几个粗定位文本行，然后处理每一个粗定位文本行
    % if size(gtRes2,1)>1        
    end %粗定位检测到多个文本行的情况
% if size(gtRes2,1)==1    
end %粗定位只检测到一个文本行的情况
end