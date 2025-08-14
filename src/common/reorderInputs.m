function inputs_veced = reorderInputs(inputs, dim1, dim2, dim3_orig, LANE_NUM, ALIGN_SIZE)
    
    dim3_gp_num = ceil(dim3_orig / LANE_NUM);
   
    dim3_aligned = dim3_gp_num * ALIGN_SIZE;

    inputs_veced = zeros(dim1, dim2, dim3_aligned);

    for n = 0:dim3_gp_num-1
        for i = 0:dim2-1
            for j = 0:dim1-1
                for k = 0:LANE_NUM-1
                    
                    if n*LANE_NUM+k < dim3_orig
                        inputs_veced(n*ALIGN_SIZE*dim2*dim1 + i*dim1*ALIGN_SIZE + j*ALIGN_SIZE + k + 1) ...
                            = inputs((n*LANE_NUM+k)*dim2*dim1 + i*dim1 + j + 1);
                    end
                end
            end
        end
    end
end