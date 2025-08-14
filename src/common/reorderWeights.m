% function weight_veced = reorderWeights(weights, dim1, dim2, dim3_orig, dim4_orig, offset, VEC_SIZE, LANE_NUM, ALIGN_SIZE, DataInType, MathType)
function weight_veced = reorderWeights(weights, dim1, dim2, dim3_orig, dim4_orig, offset, VEC_SIZE, LANE_NUM, ALIGN_SIZE)

    dim3_gp_num  = ceil(dim3_orig / VEC_SIZE);
    dim3_veced   = dim3_gp_num * VEC_SIZE;
    dim4_gp_num  = ceil(dim4_orig / LANE_NUM);
    dim4_aligned = dim4_gp_num * ALIGN_SIZE;
    
    weight_veced = zeros(dim1, dim2, dim3_veced, dim4_aligned);
    % weight_veced = fi(zeros(dim1, dim2, dim3_veced, dim4_aligned), DataInType, MathType);
    
    for m = 0:dim4_gp_num-1
        for n = 0:dim3_gp_num-1
            for i = 0:dim2-1
                for j = 0:dim1-1
                    for ll = 0:LANE_NUM-1
                        for vv = 0:VEC_SIZE-1
                            if n*VEC_SIZE+vv < dim3_orig && m*LANE_NUM+ll < dim4_orig
                            
                                orig_idx = offset + (m*LANE_NUM+ll)*dim3_orig*dim2*dim1 + ...
                                           (n*VEC_SIZE+vv)*dim2*dim1 + i*dim1 + j + 1;
                                
                                % vec_idx = sub2ind([dim1, dim2, dim3_veced, dim4_aligned], ...
                                %                   j, i, (n-1)*VEC_SIZE*ALIGN_SIZE + vv, ...
                                %                   (m-1)*dim3_veced*dim2*dim1*ALIGN_SIZE + ll);
                                vec_idx = m*dim3_veced*dim2*dim1*ALIGN_SIZE + n*dim2*dim1*VEC_SIZE*ALIGN_SIZE + ...
                                            i*dim1*VEC_SIZE*ALIGN_SIZE + j*VEC_SIZE*ALIGN_SIZE + ll*VEC_SIZE + vv + 1;
                                
                                weight_veced(vec_idx) = weights(orig_idx);
                            end
                        end
                    end
                end
            end
        end
    end
end