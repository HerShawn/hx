
% ��main�����У����neumann��contour
%������û�м�⵽bboxes������Ҫʹ��edgebox
%�ļ����

%12/18
%dict&wra�ļ����f=46.2%,��Ҫ��һ����edgebox����ֶ�λbboxes�У���
%���õ�Ч��

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
    %�ƻ���������������neumann_countour�ļ����Ϊ������һ��bboexʱ
    %��Ҫ�ں�edgebox�Ľ������ʱѡ��n_c��e_b�нϴ�ģ������غ�60%����
    %Ϊ���ս����
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