%206/2/24

clc
clear
addpath('F:\Program Files\matlab');
addpath(genpath(pwd));
do_dir='D:\hx\edgebox-contour-neumann\';
dir_img = dir([do_dir 'contour_2011train_detection\*.txt'] );
num_img = length(dir_img);
for indexImg = 1:num_img
    img_value = dir_img(indexImg).name;
    img_value = img_value(1:end-4); 
    img_name = [do_dir 'train-textloc\' img_value '.jpg'];
    g = imread(img_name);
    txt_name = [do_dir 'contour_2011train_detection\' img_value '.txt'];
    txt_name2=[do_dir 'neumann_2011train-detection2\' img_value '.txt'];
    fid = fopen(txt_name);
    txt_data = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
    dt_ct.left=txt_data{:,1};
    dt_ct.top =txt_data{:,2};
    dt_ct.right=txt_data{:,3};
    dt_ct.down=txt_data{:,4};
    dt_contour = [max(dt_ct.left,1) max(dt_ct.top,1) dt_ct.right-dt_ct.left  dt_ct.down-dt_ct.top ];
    
    
%     %   2.抑制掉宽高比<1.5的bboxes  
%     idx_contour=zeros(size(dt_contour,1),1);
%     for i=1:size(dt_contour,1)
%         if dt_contour(i,4)/dt_contour(i,3)>1.5
%             idx_contour(i,:)=1;
%         end
%     end
%     %     
%     dt_contour(find(idx_contour),:)=[];
    
    
    
    fid = fopen(txt_name2);
    txt_data2 = textscan(fid,'%d,%d,%d,%d');
    fclose(fid);
    dt_nm.left=txt_data2{:,1};
    dt_nm.top =txt_data2{:,2};
    dt_nm.right=txt_data2{:,3};
    dt_nm.down=txt_data2{:,4};
    dt_neumann=[max(dt_nm.left,1) max(dt_nm.top,1) dt_nm.right-dt_nm.left  dt_nm.down-dt_nm.top ];
    
    
    
    area_nm = dt_neumann(:,3).*dt_neumann(:,4);
    area_ct=dt_contour(:,3).*dt_contour(:,4); 
    %12/18改进：将抑制contour，变为抑制neumann
    idx_neumann=zeros(size(dt_neumann,1),1);
    idx_contour=zeros(size(dt_contour,1),1);
    
    idx_overlap=zeros(size(dt_contour,1),size(dt_neumann,1));
    
    for i=1:size(dt_contour,1)
        for j=1:size(dt_neumann,1)
            int_area = rectint(dt_neumann(j,:), dt_contour(i,:))';
            
             if double(int_area) / double(area_nm(j))>0.8
%                 idx_neumann(j,:)=1;
                  idx_overlap(i,j)=1;
            end
        end
        if(length(find(idx_overlap(i,:))))<3
            idx_neumann(j,:)=1;
        else 
             idx_contour(i,:)=1;   
        end
    end 
    
    
    dt_neumann(find(idx_neumann),:)=[];
    dt_contour(find(idx_contour),:)=[];
    
    
%     for i=1:size(dt_neumann,1)
%         for j=1:size(dt_contour,1)     
%             if abs(dt_neumann(i,2)-dt_contour(j,2))<10
%                 dt_neumann(i,:)=[min(dt_neumann(i,1),dt_contour(j,1)),min(dt_neumann(i,2),dt_contour(j,2)),max(dt_neumann(i,1)+dt_neumann(i,3),dt_contour(j,1)+dt_contour(j,3))-min(dt_neumann(i,1),dt_contour(j,1)),max(dt_neumann(i,4),dt_contour(j,4))];
%             end
%         end
%     end 
    
    
    
    dt_nc=[dt_neumann;dt_contour];
    dt_area=dt_nc(:,3).*dt_nc(:,4);
    [~,idx]=sort(dt_area,'descend');
    dt_nc=dt_nc(idx,:); 
    idx_nc=zeros(size(dt_nc,1),1);
    for i=1:size(dt_nc,1)
        for j=i+1:size(dt_nc,1)
            int_area = rectint(dt_nc(i,:), dt_nc(j,:))';
            if double(int_area) / double(dt_nc(j,3)*dt_nc(j,4))>0.6
                idx_nc(j,:)=1;
            end
        end
    end
    dt_nc(find(idx_nc),:)=[];
    
%     figure(indexImg);
%     bbGt('showRes',g,dt_nc,dt_nc);
%     save_name=[img_value '.jpg'];
%     print(indexImg, '-dpng', save_name);

          target_txt_name = [do_dir 'addContour2Neumann2\' img_value '.txt'];
%          target_txt_name=[ 'C:\Users\Administrator\Desktop\1.19日粗定位改进试验5（高宽比；重叠》0.85）\' img_value '.txt'];
          dlmwrite(target_txt_name, dt_nc);
%         dlmwrite(txt_name, txt_data,'-append');   
end