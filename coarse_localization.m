
% 在main函数中，如果neumann和contour
%方法都没有检测到bboxes，就需要使用edgebox
%的检测结果

%12/18
%dict&wra的检测结果f=46.2%,需要进一步将edgebox融入粗定位bboxes中，起到
%更好的效果

function[gt]=coarse_localization(img,gt_nc,model,opts)
bbs=edgeBoxes(img,model,opts);
    bbs=bbs([1,2,6,7],:);
    bbs=bbs(:,1:4);
    bbs(:,3)=bbs(:,1)+bbs(:,3);
    bbs(:,4)=bbs(:,2)+bbs(:,4);
    gt_eb=zeros(1,4);
    gt_eb(1,1)=min(bbs(:,1));
    gt_eb(1,2)=min(bbs(:,2));
    gt_eb(1,3)=max(bbs(:,3))-gt_eb(1,1);
    gt_eb(1,4)=max(bbs(:,4))-gt_eb(1,2);
    %计划是这样：当发现neumann_countour的检测结果为至多有一个bboex时
    %需要融合edgebox的结果。暂时选择n_c和e_b中较大的（至少重合60%）的
    %为最终结果。
    area_nc = gt_nc(:,3).*gt_nc(:,4);
    area_eb=gt_eb(:,3).*gt_eb(:,4); 
    idx_nc=zeros(size(gt_nc,1),1);
    for i=1:size(gt_eb,1)
        for j=1:size(gt_nc,1)
            int_area = rectint(gt_nc(j,:), gt_eb(i,:))';
            if double(int_area) / double(area_nc(j))>0.6
                idx_nc(j,:)=1;
            end
        end
    end 
    gt_nc(find(idx_nc),:)=[];
    gt=[gt_nc;gt_eb];
    return
end