% 12/8
function predwords=predwords_sort(img_gt,i,predwords,bboxes)
if size(predwords,2)==0
%     fprintf('TOTAL_EDIT_DISTANCE =  %d', size(img_gt,1));
else
    if(i==1)
        temp_predwords=zeros(length(strtrim(predwords)),1);
    end
    if ~isempty(bboxes)
        for idx=1:i
        temp_predwords(idx,:)=length(strtrim(predwords{idx}));
        end
        [~,I]=sort(temp_predwords,'descend');
        temp_predwords=temp_predwords(I);
        predwords=predwords(I);
    end
    
end
end