function output = my_layernorm_repmat(input, epsilon, dim, scale, shift)
    input = single(input); 

    % Calculate the mean and standard deviation of the input data
    mean_val = mean(input, dim);

    diff = input - mean_val;

    diff_sq = diff.^2; 
    
    mean_diff_sq = mean(diff_sq, dim);
    
    % save('test.mat', 'mean_diff_sq');
    std_val = sqrt(single(mean_diff_sq)); 

    % Calculate the normalized data
    normalized = (input - mean_val) ./ (std_val + epsilon);
    
%     disp(normalized);
%     disp(size(normalized));

    % Apply scaling and offset
    scale = repmat(scale, [size(input, 2), 1]);
    scale = single(permute(scale, [2, 1]));
%     disp(scale);
%     disp(size(scale));
    shift = repmat(shift, [size(input, 2), 1]);
    shift = single(permute(shift, [2, 1]));
%     disp(shift);
%     disp(size(shift));

    output = normalized .* scale + shift;
end

%% available 202307141120
% function output = my_layernorm_repmat(input, epsilon, dim, scale, shift)
%     input = single((input)); 
%     % Calculate the mean and standard deviation of the input data
%     [std_val , mean_val]= std(input, 1, dim);
% 
%     % Calculate the normalized data
%     normalized = (input - mean_val) ./ (std_val + epsilon);
% 
%     % Apply scaling and offset
%     scale = repmat(scale, [size(input, 2), 1]);
%     scale = permute(scale, [2, 1]);
% 
%     shift = repmat(shift, [size(input, 2), 1]);
%     shift = permute(shift, [2, 1]);
% 
%     output = normalized .* scale + shift;
% end
