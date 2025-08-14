function [ ] = fix_check( A , B  )
    error_count = 0;
    [Win,Hin,N] = size(A);
    A_single = single(A);
    B_single = single(B);
    
    abs_value = abs(A_single(:,:,:)-B_single(:,:,:));
    error_count = length( find( abs_value(:,:,:) > 0.0001 ));
    [max_val, position_max] = max(abs_value(:)); 
    [x,y,z] = ind2sub(size(abs_value),position_max);
    fprintf('The max_index is : [%d,%d,%d]\n',x,y,z);
    fprintf('The max_value is : %f\n',max_val);
    
    if (  max_val < 0.0001  )
        fprintf('!!!!!!!!check pass!!!!!!!!!\n\n');
    else
        fprintf(2,'A is %f\n',A(x,y,z));
        fprintf(2,'B is %f\n',B(x,y,z));
        
        fprintf(2,'error_num is %d\n',error_count);
        fprintf(2,'All_num is %d\n\n',Win*Hin*N);
    end
    
end
