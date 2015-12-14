% 2015/12/7
% ∫Û¥¶¿Ì/≤‚∆¿

function post_precess(im,numbbox1,wbboxes,predwords,thresh,img_gt)
global totalPredBbox totalGoodBbox; 
[height, width, ~] = size(im);
if numbbox1~=0 &&~isempty(wbboxes)
    %remove wbboxes with low recognition scores
    bad_idx = wbboxes(:,end)<thresh;
    wbboxes(bad_idx, :) = [];
    predwords(bad_idx) = [];
    numbbox2 = size(wbboxes,1);
    pred_taken = zeros(numbbox2,1);
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
                if taken(tt)==0 && pred_taken(bidx)==0 && strcmpi(strtrim(pred_tag), cell2mat(img_gt(tt,5)))          
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
end