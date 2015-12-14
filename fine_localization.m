
% 12.1 ����ϸ��λ
%һ��һ����:�ȸ㶨CHAR��ʶ��ģ��
%Ϊʹ���򾮾���������STR_BS2���й�ϸ��λ�Ĳ�����ֲ������
% ϸ��λ��Ϊ�����ָ�/beam search" ��ʶ�� ������
%beam search�Ĺؼ����ڵ����ָ��������߷ָ�Ч����
%ʶ��Ĺؼ����� ���CHAR��DICTģ�ͣ����ʶ��Ч����
%����ؼ����ڽ�ʶ������ʽ����ȥ���࣬��߲���Ч����

function fine_localization(gtRes2,g)
addpath(genpath('../detectorDemo'));
run model_release/matconvnet/matlab/vl_setupnn.m
[height,width,~] = size(g);
%����ֶ�λ��Ľ��ֻ��һ��bounding box����ôֱ����ʶ���������������ֵ����һ��ֶ�λ��CNN��edgebox?)
% �ֶ�λֻ�õ�һ��bboxʱ������beam search/�ָ��Ҫ��ʶ��ͺ���
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
%ϸ��λCNN�����
    if size(gtRes2,1)>1
        %���ڴֶ�λ�õ���ÿһ���ı��У�
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
            % �Ӵֶ�λ�ı�������CNN����ӽ���ϸ��λ��          
            for bidx = 1:numbbox
                %ϸ��λ�Ĺ��˲��裺
                if bboxes(bidx,5)>0.8 && length(spaces(bidx).locations)<5   
                %  beam search/�ָ������Ϳ�ʼ��                
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
                    %  ����ֻ�����CHAR��һ��ģ�͵�ʶ�𣬰�����beam search��                       
                        [newstates ,curr]= beam_search(im,states,  curr, segs, spacescores, numbeams, thresh,c_split);
                        states = newstates;
                    end
                    fprintf('CHAR prediction: ')
                    states{1}
                    
                %if bboxes(bidx,5)>0.8 && length(spaces(bidx).locations)<5                     
                end   %���ڲ�
            % for bidx = 1:numbbox               
            end %һ���ֶ�λ�ı����ڣ���⵽����ϸ��λbboxes��Ȼ����ÿһ��bbox
        % for i=1:size(gtRes2,1)            
        end %���м����ֶ�λ�ı��У�Ȼ����ÿһ���ֶ�λ�ı���
    % if size(gtRes2,1)>1        
    end %�ֶ�λ��⵽����ı��е����
% if size(gtRes2,1)==1    
end %�ֶ�λֻ��⵽һ���ı��е����
end