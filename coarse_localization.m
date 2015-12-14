
% 在main函数中，如果neumann和contour
%方法都没有检测到bboxes，就需要使用edgebox
%的检测结果

function[gt]=coarse_localization(img,model,opts)
bbs=edgeBoxes(img,model,opts);
    bbs=bbs([1,2,6,7],:);
    bbs=bbs(:,1:4);
    bbs(:,3)=bbs(:,1)+bbs(:,3);
    bbs(:,4)=bbs(:,2)+bbs(:,4);
    gt=zeros(1,4);
    gt(1,1)=min(bbs(:,1));
    gt(1,2)=min(bbs(:,2));
    gt(1,3)=max(bbs(:,3))-gt(1,1);
    gt(1,4)=max(bbs(:,4))-gt(1,2);
    return
end