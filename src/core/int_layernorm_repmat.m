function output = int_layernorm_repmat(input, epsilon, dim, scale, shift)
% function output = fix_ew_LN_lane(input, scale, shift)

% LANEDataOutPath = "./mat_output/ln_lane_pc/";
MathType   = fimath('RoundingMethod', 'Nearest', 'OverflowAction', 'Saturate', 'ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');

LN_I_input_Width = 64;
LN_I_input_Frac  = 0.0;
LN_I_input_Type{1} = numerictype('Signed',1, 'WordLength', LN_I_input_Width, 'FractionLength', LN_I_input_Frac);    

LN_I_x_mean_round_Width = 64;
LN_I_x_mean_round_Frac  = 0.0;
LN_I_x_mean_round_Type{1} = numerictype('Signed',1, 'WordLength', LN_I_x_mean_round_Width, 'FractionLength', LN_I_x_mean_round_Frac);    

LN_I_x2_mean_round_Width = 64;
LN_I_x2_mean_round_Frac  = 0.0;
LN_I_x2_mean_round_Type{1} = numerictype('Signed',1, 'WordLength', LN_I_x2_mean_round_Width, 'FractionLength', LN_I_x2_mean_round_Frac);    

LN_I_sqrt_output_round_Width = 64;
LN_I_sqrt_output_round_Frac  = 0.0;
LN_I_sqrt_output_round_Type{1} = numerictype('Signed',1, 'WordLength', LN_I_sqrt_output_round_Width, 'FractionLength', LN_I_sqrt_output_round_Frac);    

LN_I_normalized_x_q_M_cvalue_round_Width = 64;
LN_I_normalized_x_q_M_cvalue_round_Frac  = 0.0;
LN_I_normalized_x_q_M_cvalue_round_Type{1} = numerictype('Signed',1, 'WordLength', LN_I_normalized_x_q_M_cvalue_round_Width, 'FractionLength', LN_I_normalized_x_q_M_cvalue_round_Frac);    


    channel_nums = 192;

    x_q = single(input);

    %%  x_q = (x * 64).round()  
    % x_q = (x_q * 128);
    x_q = (x_q * 64);
    % x_q = (x_q * 32);

    I_x_q = fi(x_q,LN_I_input_Type{1}, MathType);  % round
    I_x_q = single(I_x_q);

    % p_qudq_blocks_0_xM64 = load('./mat_testdata/cpu_p_qudq_0_LN_xM64_fl_-3.mat'); 
    % p_qudq_blocks_0_xM64_2dim = reshape(p_qudq_blocks_0_xM64.data,size(p_qudq_blocks_0_xM64.data,2),size(p_qudq_blocks_0_xM64.data,3));
    % disp('I_xM64');
    % fix_check(p_qudq_blocks_0_xM64_2dim,I_x_q);

    %%  x_q_fdim_sum = (x_q.sum(dim=-1)) 
    % I_x_q_fdim_sum = sum(I_x_q, 2); % dim = 2
    I_x_q_fdim_sum = sum(I_x_q, dim); % dim = 2
    
    % % % save x data   197*192
    % fi_I_x_q_fdim_sum = fi(I_x_q_fdim_sum,LN_I_input_Type{1}, MathType);  % round
    % fid = fopen(strcat(LANEDataOutPath,'I_x_q_fdim_sum.data'), 'w');
    % fwrite(fid, storedInteger(fi_I_x_q_fdim_sum), 'int32');
    % fclose(fid);

    %%  x_q_fdim_mean = (x_q_fdim_sum / channel_nums).round() 
    x_q_fdim_mean = (I_x_q_fdim_sum ./ channel_nums);

    fi_I_x_q_fdim_mean = fi(x_q_fdim_mean, LN_I_x_mean_round_Type{1}, MathType); % round
    I_x_q_fdim_mean = single(fi_I_x_q_fdim_mean);

    %%  x_q_squared_fdim_sum = (((x_q**2).sum(dim=-1))) 
    % I_x_q_squared_fdim_sum = sum((I_x_q.^2), 2); % dim = 2 % .^
    I_x_q_squared_fdim_sum = sum((I_x_q.^2), dim); % dim = 2 % .^
    
    % % % save x data   197*192
    % fi_I_x_q_squared_fdim_sum = fi(I_x_q_squared_fdim_sum,LN_I_input_Type{1}, MathType);  % round
    % fid = fopen(strcat(LANEDataOutPath,'I_x_q_squared_fdim_sum.data'), 'w');
    % fwrite(fid, storedInteger(fi_I_x_q_squared_fdim_sum), 'int32');
    % fclose(fid);

    %%  x_q_squared_fdim_mean = (x_q_squared_fdim_sum / channel_nums).round() 
    x_q_squared_fdim_mean = (I_x_q_squared_fdim_sum ./ channel_nums);

    I_x_q_squared_fdim_mean = fi(x_q_squared_fdim_mean, LN_I_x2_mean_round_Type{1}, MathType); % round
    I_x_q_squared_fdim_mean = single(I_x_q_squared_fdim_mean);

    %%  value_token_root_of =  (x_q_squared_fdim_mean - x_q_fdim_mean**2)  
    I_value_token_root_of =  (I_x_q_squared_fdim_mean - I_x_q_fdim_mean.^2); % .^

    %%  sqrt_value_token_root_of = torch.sqrt(value_token_root_of).round() 
    sqrt_value_token_root_of = sqrt(single(I_value_token_root_of));

    fi_I_sqrt_value_token_root_of = fi(sqrt_value_token_root_of, LN_I_sqrt_output_round_Type{1}, MathType); % round
    I_sqrt_value_token_root_of = single(fi_I_sqrt_value_token_root_of);

    %%  mean_x_q =(x_q_fdim_mean) 
    I_mean_x_q = I_x_q_fdim_mean;

    % p_qudq_blocks_0_mean = load('./mat_testdata/cpu_p_qudq_0_LN_mean_fl_5.mat'); 
    % p_qudq_blocks_0_mean_permute = permute(p_qudq_blocks_0_mean.data, [2, 1]);
    % disp('I_mean');
    % fix_check(p_qudq_blocks_0_mean_permute,I_mean_x_q);
  
    %%  std_x_q = (sqrt_value_token_root_of)
    I_std_x_q = I_sqrt_value_token_root_of;

    % p_qudq_blocks_0_std = load('./mat_testdata/cpu_p_qudq_0_LN_std_fl_0.mat'); 
    % p_qudq_blocks_0_std_permute = permute(p_qudq_blocks_0_std.data, [2, 1]);
    % disp('I_std');
    % fix_check(p_qudq_blocks_0_std_permute,I_std_x_q);
   

    %%  x_q = ((((x_q - mean_x_q.unsqueeze(-1))*2048 / std_x_q.unsqueeze(-1))).round()) / 2048  # 4 = 16 - 1 - 11 # 2^11 = 2048
    normalized_x_q_M_cvalue = ((I_x_q - I_mean_x_q)*2048) ./ I_std_x_q;  % 2^11 = 2048

    I_normalized_x_q_M_cvalue = fi(normalized_x_q_M_cvalue, LN_I_normalized_x_q_M_cvalue_round_Type{1}, MathType); % round
    I_normalized_x_q_M_cvalue = single(I_normalized_x_q_M_cvalue);

    %% I_normalized_x_q_M_cvalue.data
    % LN_LANE_NUM = 8;
    % LN_ALIGN_SIZE = 8;
    % I_normalized_x_q_M_cvalue_3dim = reshape(I_normalized_x_q_M_cvalue,1,size(I_normalized_x_q_M_cvalue,1),size(I_normalized_x_q_M_cvalue,2));
    % rI_I_normalized_x_q_M_cvalue_3dim = reorderInputs(I_normalized_x_q_M_cvalue_3dim, 1, 197, 192, LN_LANE_NUM, LN_ALIGN_SIZE);
    % rI_I_normalized_x_q_M_cvalue_2dim = reshape(rI_I_normalized_x_q_M_cvalue_3dim,size(rI_I_normalized_x_q_M_cvalue_3dim,2),size(rI_I_normalized_x_q_M_cvalue_3dim,3));
    % fi_rI_I_normalized_x_q_M_cvalue_2dim = fi(rI_I_normalized_x_q_M_cvalue_2dim,LN_I_normalized_x_q_M_cvalue_round_Type{1}, MathType); 
    % % reorderInputs_norm2_input_2dimInteger = storedInteger(fi_reorderInputs_norm2_input_2dim);
    % 
    % % % save x data   197*192
    % fid = fopen(strcat(LANEDataOutPath,'lane_I_normalized_x_q_M_cvalue.data'), 'w');
    % fwrite(fid, storedInteger(fi_rI_I_normalized_x_q_M_cvalue_2dim), 'int32');
    % fclose(fid);
    % 
    % % % save x data   197*192
    % fid = fopen(strcat(LANEDataOutPath,'I_mean_x_q.data'), 'w');
    % fwrite(fid, storedInteger(fi_I_x_q_fdim_mean), 'int16');
    % fclose(fid);
    % 
    % % % save x data   197*192
    % fid = fopen(strcat(LANEDataOutPath,'I_std_x_q.data'), 'w');
    % fwrite(fid, storedInteger(fi_I_sqrt_value_token_root_of), 'int16');
    % fclose(fid);
    %%
    % normalized_x_q = I_normalized_x_q_M_cvalue / 2048;
    normalized_x_q = I_normalized_x_q_M_cvalue * 0.00048828125;  % 2^11 = 2048  % 1 / 2048 = 0.00048828125

    %% x = x_q * weight.reshape(1,1,-1) + bias.reshape(1,1,-1)
    % scale = single(scale);
    scale = repmat(scale, [size(input, 2), 1]);
    scale = single(permute(scale, [2, 1]));

    % shift = single(shift);
    shift = repmat(shift, [size(input, 2), 1]);
    shift = single(permute(shift, [2, 1]));

    output = normalized_x_q .* scale + shift;
    
end