function output_weight = vectorizeWeight(input_weight, VEC_SIZE, LANE_NUM)
% vectorize the weight with the giving parmeters VEC_SIZE and LANE_NUM
% vectorized the weights, ALIGN_SIZE*VEC_SIZE bytes are packed together(DDR II), but only the first LANE_NUM*VEC_SIZE bytes are valid.
% input_weight = storedInteger(input_weight);
input_weight = permute(input_weight, [2,1,3,4]); %
[H,W,N,M]=size(input_weight);%height weight N and M of the weight kernels

ALIGN_SIZE   = 2^ceil(log2(LANE_NUM));
dim3_gp_num  = ceil(N/VEC_SIZE);
dim3_veced   = dim3_gp_num * VEC_SIZE;
dim4_gp_num  = ceil(M/LANE_NUM);
dim4_aligned = dim4_gp_num * ALIGN_SIZE;

MathType      = fimath('RoundingMethod', 'Nearest', 'OverflowAction', 'Saturate', 'ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');
WeightType    = numerictype('Signed',input_weight.Signed, 'WordLength', input_weight.WordLength, 'FractionLength', input_weight.FractionLength);
output_weight = zeros(1,H*W*dim3_veced*dim4_aligned);
% output_weight = fi(output_weight, WeightType, MathType);
input_weight  = single(input_weight); 

for m=0:dim4_gp_num-1
    for n=0:dim3_gp_num-1
        for i=0:H-1
            for j=0:W-1
                for ll=0:LANE_NUM-1
                    for vv=0:VEC_SIZE-1
                        if n*VEC_SIZE+vv<N && m*LANE_NUM+ll<M
                            output_weight(m*dim3_veced*H*W*ALIGN_SIZE + n*H*W*VEC_SIZE*ALIGN_SIZE + i*W*VEC_SIZE*ALIGN_SIZE + j*VEC_SIZE*ALIGN_SIZE + ll*VEC_SIZE + vv + 1)= ...,
                            input_weight(i+1, j+1, n*VEC_SIZE+vv+1, m*LANE_NUM+ll+1);
                        end
                    end
                end
            end
        end
    end
end

% txt_fp = fopen("D:\rundir\matlab_cnn_fix\bb.txt", 'a+');
% for i=1:H*W*dim3_veced*dim4_aligned
%     fprintf(txt_fp, "%d\n", storedInteger(output_weight(i)));
% end
% fclose(txt_fp);

output_weight = fi(output_weight, WeightType, MathType);

end

