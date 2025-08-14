%% Calculate the top_k accuracy
function [numBetterProb, accuracy] = getAccuracy_pytorch(searchTop, label, current_accuracy, Batch_idx, LogFid, Iter, BatchSize, FirstImgIdx)

% Scheme-1 same as pytorch
accuracy = current_accuracy;
predictTop5 = zeros(5,1);
maxNum = 0;
for k=1:5
	if (maxNum==0)
		predictLabel = find(searchTop==max(searchTop));
		maxNum = length(predictLabel); % in case multiple label share the same max prob value
	end
	predictTop5(k) = predictLabel(maxNum)-1; % Note the label in val.txt start from 0
    searchTop(predictLabel(maxNum)) = -Inf; % remove top value
	maxNum = maxNum - 1;
end
% Count top-1 accuracy
if(predictTop5(1)==label(Batch_idx))
   accuracy(1) = accuracy(1)+1;
end
% Count top-5 accuracy
if(isempty(find(predictTop5==label(Batch_idx), 1))==0)
   accuracy(2) = accuracy(2)+1;
end
numBetterProb = 0; %in getAccuracy_pytorch method, this value is not used
% write to log file
fprintf(LogFid, 'Pic = %d Prob = %.2f %.2f %.2f %.2f %.2f \n\n', Iter*BatchSize+Batch_idx+FirstImgIdx, predictTop5(1), predictTop5(2), predictTop5(3), predictTop5(4), predictTop5(5));

%% Scheme-2 (same as Caffe)
% accuracy = current_accuracy; % top_1, top_5
% 
% labelBetterProb = find(searchTop>=searchTop(label(Batch_idx)+1)); % label value starts from 0
% numBetterProb = length(labelBetterProb); % for loging
% % Count top-1 accuracy
% if(length(labelBetterProb)<=1)
%     accuracy(1)=accuracy(1)+1;
% end
% % Count top-5 accuracy
% if(length(labelBetterProb)<=5)
%     accuracy(2)=accuracy(2)+1;
% end
