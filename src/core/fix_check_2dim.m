function [ ] = fix_check( A , B )
    error_count = 0;
    [Win,Hin] = size(A);
    A_single = single(A);
    B_single = single(B);
    
    abs_value = abs(A_single(:,:) - B_single(:,:));
    
    different_elements = find(abs_value > 0.0001);
    error_count = length(different_elements);
    
    for i = 1:error_count
        [x, y] = ind2sub(size(abs_value), different_elements(i));
        fprintf('Error %d:\n', i);
        fprintf('  Index: [%d, %d]\n', x, y);
        fprintf('  A: %f\n', A(x, y));
        fprintf('  B: %f\n', B(x, y));
        fprintf('  Difference: %f\n', abs_value(x, y));
        fprintf('\n');
    end
    
    [max_val, position_max] = max(abs_value(:)); 
    [x, y] = ind2sub(size(abs_value), position_max);
    fprintf('The max_index is : [%d, %d]\n', x, y);
    fprintf('The max_value is : %f\n', max_val);
    
    if (max_val < 0.0001)
        fprintf('!!!!!!!!check pass!!!!!!!!!\n\n');
    else
        fprintf(2,'error_num is %d\n', error_count);
        fprintf(2,'All_num is %d\n\n', Win * Hin);
    end
end