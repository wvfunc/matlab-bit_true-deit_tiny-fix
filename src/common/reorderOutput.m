function output_reorder = reorderOutput(output, dim1, dim2, dim3_orig, LANE_NUM, ALIGN_SIZE)
% Reorder output array
dim3_gp_num = ceil(dim3_orig/LANE_NUM);
% output_reorder = zeros(dim1*dim2*dim3_gp_num*LANE_NUM, 1);
output_reorder = zeros(dim1* dim2, dim3_orig);

    for k = 0:dim3_gp_num-1
        for i = 0:dim2-1
            for j = 0:dim1-1
                for vv = 0:LANE_NUM-1
                    if k*LANE_NUM+vv < dim3_orig
                        output_reorder((k*LANE_NUM+vv)*dim2*dim1 + i*dim1 + j + 1) ...
                            = output(k*dim2*dim1*ALIGN_SIZE + i*dim1*ALIGN_SIZE + j*ALIGN_SIZE + vv+1);
                    end
                end
            end
        end
    end

end
