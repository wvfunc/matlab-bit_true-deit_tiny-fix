function output_bias = vectorizeBias(input_bias, LANE_NUM)
% vectorize the weight with the giving parmeters VEC_SIZE and LANE_NUM
% vectorized the weights, ALIGN_SIZE*VEC_SIZE bytes are packed together(DDR II), but only the first LANE_NUM*VEC_SIZE bytes are valid.
% input_weight = storedInteger(input_weight);
[M,~]=size(input_bias);

ALIGN_SIZE   = 2^ceil(log2(LANE_NUM));
dim4_gp_num  = ceil(M/LANE_NUM);
dim4_aligned = dim4_gp_num * ALIGN_SIZE;

MathType    = fimath('RoundingMethod', 'Nearest', 'OverflowAction', 'Saturate', 'ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');
BiasType    = numerictype('Signed',input_bias.Signed, 'WordLength', input_bias.WordLength, 'FractionLength', input_bias.FractionLength);
output_bias = zeros(dim4_aligned,1);
input_bias  = single(input_bias);

for m=0:dim4_gp_num-1
    for ll=0:LANE_NUM-1
        if m*LANE_NUM + ll<M
            output_bias(m*ALIGN_SIZE + ll + 1) = input_bias(m*LANE_NUM + ll + 1);
        end
    end
end

% txt_fp = fopen("D:\rundir\matlab_cnn_fix\cc.txt", 'a+');
% for i=1:dim4_aligned
%     fprintf(txt_fp, "%d\n", storedInteger(output_bias(i)));
% end
% fclose(txt_fp);

output_bias = fi(output_bias, BiasType, MathType);

end

