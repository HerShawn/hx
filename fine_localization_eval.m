%2015/12/18
%查看下粗定位测评成绩
function fine_localization_eval(lc_es,indexImg)
% clear
% clc

% dir_gt = dir('E:\2012 文字检测\测试集\ICDAR 2011\test-textloc-gt\*.txt');
% dir_gt = dir('G:\数据\icdar2011\test-textloc\*.txt');
% dir_es = dir('E:\2013毕设文字检测\试验结果\一起训练\location13\*.txt');
dir_es = dir('D:\hx\edgebox-contour-neumann\coarse_localization2\*.txt');
num_img = length(dir_es);
p_each = zeros(1,num_img);
r_each = zeros(1,num_img);
num_gt_rect = 0;
value_pr_rect = 0;
num_es_rect = 0;
value_rc_rect = 0;
tr = 0.8;
tp = 0.4;
% max_t = zeros(5,12);

       
%                  disp(index)
%             gt_name = ['E:\2012 文字检测\测试集\ICDAR 2011\test-textloc-gt\gt_' dir_es(index).name];
            gt_name = ['D:\hx\edgebox-contour-neumann\train-textloc\gt_' dir_es(indexImg).name];
%             es_name = ['E:\2013毕设文字检测\试验结果\一起训练\location13\' dir_es(index).name];
%             es_name = ['D:\hx\edgebox-contour-neumann\coarse_localization2\' dir_es(index).name];
            % 读groundtruth坐标
            fid = fopen(gt_name);
            txt_data = textscan(fid,'%d,%d,%d,%d,%s');
            fclose(fid);
            num_gt = length(txt_data{2});
            lc_gt = zeros(num_gt,4);
            for i = 1:num_gt
                lc_gt(i,1) = txt_data{1}(i);
                lc_gt(i,2) = txt_data{2}(i);
                lc_gt(i,3) = txt_data{3}(i);
                lc_gt(i,4) = txt_data{4}(i);
            end
            %     lc_gt = dlmread(gt_name);
            % 读估计坐标
            %     fid = fopen(es_name);
            %     txt_data = textscan(fid,'%d,%d,%d,%d');
            %     if ~~isempty(txt_data{1})
            %         fclose(fid);
            %         continue
            %     end
            %     lc_es =  dlmread(es_name);
%             fid = fopen(es_name);
%             txt_data = textscan(fid,'%d,%d,%d,%d');
%             fclose(fid);
%             num_es = length(txt_data{2});
%             lc_es = zeros(num_es,4);
%             for i = 1:num_es
%                 lc_es(i,1) = txt_data{1}(i);
%                 lc_es(i,2) = txt_data{2}(i);
%                 lc_es(i,3) = txt_data{3}(i)+txt_data{1}(i);
%                 lc_es(i,4) = txt_data{4}(i)+txt_data{2}(i);
%             end
            num_gt = size(lc_gt,1);
            num_es = size(lc_es,1);
            pm = zeros(num_gt,num_es);   %precision 矩阵
            rm = zeros(num_gt,num_es);   %recall 矩阵
            for i = 1:num_gt
                for j = 1:num_es
                    intersection_left = max(lc_es(j,1),lc_gt(i,1));
                    intersection_top = max(lc_es(j,2),lc_gt(i,2));
                    intersection_right = min(lc_es(j,3),lc_gt(i,3));
                    intersection_bottom = min(lc_es(j,4),lc_gt(i,4));
                    num_intersection = max(intersection_right-intersection_left+1,0)*max(intersection_bottom-intersection_top+1,0);
                    num_pixel_est = (lc_es(j,3)-lc_es(j,1)+1)*(lc_es(j,4)-lc_es(j,2)+1);
                    num_pixel_gt = (lc_gt(i,3)-lc_gt(i,1)+1)*(lc_gt(i,4)-lc_gt(i,2)+1);
                    pm(i,j) = num_intersection/num_pixel_est;
                    rm(i,j) = num_intersection/num_pixel_gt;
                end
            end
            % 计算查准率
            num_es_rect = num_es_rect + num_es;
            value_each_p = 0;
            for i = 1:num_es
                rm_vec = rm(:,i);
                pm_vec = pm(:,i);
                flag_pre_one = rm_vec>tr;
                num_pre_one = sum(flag_pre_one);
                
                if num_pre_one > 1
                    value_each_p = value_each_p + 0.8*double(sum(pm_vec(flag_pre_one))>tp);
                else
                    [~,index_des] = max(pm_vec);
                    %             value_each_p = value_each_p + double(pm_vec(index_des)>tp);
                    %             if pm_vec(index_des)>tp&&rm(index_des,i)<tr
                    %             end
                    %             value_each_p = value_each_p + double(pm_vec(index_des)>tp);
                    value_each_p = value_each_p + double(pm_vec(index_des)>tp&&sum(rm(index_des,:))>tr);
                    
                end
            end
            p_each(indexImg) = value_each_p/num_es;
            value_pr_rect = value_pr_rect + value_each_p;
            % 计算查全率
            num_gt_rect = num_gt_rect + num_gt;
            if num_es == 0
                return;
            end
            value_each_r = 0;
            for i = 1:num_gt
                pm_vec = pm(i,:);
                rm_vec = rm(i,:);
                flag_rec_one = pm_vec>tp;
                num_rec_one = sum(flag_rec_one);
                
                if num_rec_one > 1
                    value_each_r = value_each_r + 0.8*double(sum(rm_vec(flag_rec_one))>tr);
                else
                    [~,index_des] = max(rm_vec);
                    value_each_r = value_each_r + double(rm_vec(index_des)>tr&&sum(pm(:,index_des))>tp);
                end
            end
            r_each(indexImg) = value_each_r/num_gt;
            value_rc_rect = value_rc_rect + value_each_r;
       
        fine_localization_recall = value_rc_rect/num_gt_rect
        fine_localization_precision = value_pr_rect/num_es_rect
        fine_localization_f = 2/(1/fine_localization_recall+1/fine_localization_precision)
%          disp([recall  precision f])
%         index_x = int32((tr - 0.3)*10);
%         index_y = int32((tp - 0.3)*10);
%         max_t(index_y,(index_x - 1)*3 + 1) = precision;
%         max_t(index_y,(index_x - 1)*3 + 2) = recall;
%         max_t(index_y,(index_x - 1)*3 + 3) = f;
%     end
% end
% r_each1 = r_each;
% p_each1 = p_each;
% save eachdata.mat r_each1 p_each1
end