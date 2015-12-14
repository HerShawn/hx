% 12/8
function char_edit_distance(img_gt,predwords)
global total_edit_distance;
max_dist=0;
assignment_mat=ones(size(img_gt,1),length(predwords))*99;

for ed_i=1:size(img_gt,1)
    for ed_j=1:length(predwords)
        assignment_mat(ed_i,ed_j) = EditDistance(lower(cell2mat(img_gt(ed_i,5))),strtrim(cell2mat(predwords(ed_j))));
        max_dist = max(max_dist,assignment_mat(ed_i,ed_j));
    end
end

words_detection_matched=cell(1);

for search_dist=0: max_dist
    ii=0;
    while ii<size(assignment_mat,1)
        ii=ii+1;
        [~,min_dist_idx]=min(assignment_mat(ii,:));
        if assignment_mat(ii,min_dist_idx) == search_dist
            total_edit_distance= total_edit_distance+assignment_mat(ii,min_dist_idx);
            words_detection_matched(end+1)={min_dist_idx};
            img_gt(ii,:)=[];
            assignment_mat(ii,:)=[];
            for jj=1:size(assignment_mat,1)
                assignment_mat(jj,min_dist_idx)=99;
            end
            ii=ii-1;
        end
    end
end

for  i2=1: size(img_gt,1)
    fprintf('检测到多个bbox时 GT word \"%s\" no match found\n', cell2mat(img_gt(i2,5)));
    total_edit_distance=total_edit_distance+length(cell2mat(img_gt(i2,5)))
end

for j2=1:length(predwords)
    
    %                 if (find(words_detection_matched.begin(),words_detection_matched.end(),j) == words_detection_matched.end())
    if  isempty(find(cell2mat(words_detection_matched(:))==[j2]))
        %                     //cout << " Detection word \"" << words_detection[j] << "\" no match found" << endl;
        fprintf('检测到多个bbox时 DT word \"%s\" no match found\n', cell2mat(words_detection_matched(j2)));
        total_edit_distance=total_edit_distance+ length(cell2mat(predwords(j2)))
    end
end

end