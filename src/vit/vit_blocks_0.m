function result = vit_blocks_0(input)
    global Save;
    global IsCheck;
    global MatOutPath;
    global OFFLINE_REORDER;
    global VEC_SIZE;
    global LANE_NUM;
    
    % Define the folder name
    folderName = 'blocks_0';
    
    % Create the folder
    if ~exist(fullfile(MatOutPath, folderName), 'dir')
        mkdir(fullfile(MatOutPath, folderName));
    end

    %% input
    qudq_qudq_x_And_qudq_pos = input;

    %% MathType
    
    MathType   = fimath('RoundingMethod', 'Nearest', 'OverflowAction', 'Saturate', 'ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');
   
    %% blocks_0_LayerNorm_1
    
    % Getting parameters %
    % blocks_0_norm1_weight = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_norm1_weight_fl_6.mat'); 
    % blocks_0_norm1_bias = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_norm1_bias_fl_7.mat'); 

    blocks_0_norm1_weight = load('./mat_extract/deit_tiny/blocks/block_0/cpu_p_qudq_0_LN_weight_fl_6.mat'); 
    blocks_0_norm1_bias = load('./mat_extract/deit_tiny/blocks/block_0/cpu_p_qudq_0_LN_bias_fl_6.mat'); 

    blocks_0_norm1_weight = blocks_0_norm1_weight.data;
    blocks_0_norm1_bias = blocks_0_norm1_bias.data;
    
    % Fixed point number %
    blocks_0_qact1_Width = 8;
    blocks_0_qact1_Frac  = 4.0;
    
    blocks_0_qact1_Type{1}  = numerictype('Signed',1, 'WordLength', blocks_0_qact1_Width, 'FractionLength', blocks_0_qact1_Frac); 
    
    %% Attention % Linear to get qkv
    
    % Getting parameters %  
    blocks_0_attn_qkv_weight = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_attn_qkv_weight_fl_8.mat'); 
    blocks_0_attn_qkv_bias = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_attn_qkv_bias_fl_6.mat'); 
    blocks_0_attn_qkv_weight = blocks_0_attn_qkv_weight.data;
    blocks_0_attn_qkv_bias = blocks_0_attn_qkv_bias.data;
    
    % Fixed point number %
    blocks_0_attn_qkv_weight_Width = 8;
    blocks_0_attn_qkv_weight_Frac  = 8.0;
    blocks_0_attn_qkv_bias_Width = 8;
    blocks_0_attn_qkv_bias_Frac  = 6.0;
    
    blocks_0_attn_qact1_Width = 8;
    blocks_0_attn_qact1_Frac  = 4.0;
    
    blocks_0_attn_qkv_weight_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_qkv_weight_Width, 'FractionLength', blocks_0_attn_qkv_weight_Frac); 
    blocks_0_attn_qkv_bias_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_qkv_bias_Width, 'FractionLength', blocks_0_attn_qkv_bias_Frac); 
    blocks_0_attn_qact1_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_qact1_Width, 'FractionLength', blocks_0_attn_qact1_Frac); 
    
    %% Attention  % SDP
    
    % Fixed point number %
    blocks_0_attn_qact_attn1_Width = 8;
    blocks_0_attn_qact_attn1_Frac  = 3.0;
    
    blocks_0_attn_qact_attn1_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_qact_attn1_Width, 'FractionLength', blocks_0_attn_qact_attn1_Frac); 
    
    %% Attention  % softmax
    
    blocks_0_attn_log_int_softmax_Width = 8;
    blocks_0_attn_log_int_softmax_Frac  = 7.0;
    
    blocks_0_attn_log_int_softmax_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_log_int_softmax_Width, 'FractionLength', blocks_0_attn_log_int_softmax_Frac); 
    
    %% Attention  % attn @ v
    
    blocks_0_attn_qact2_Width = 8;
    blocks_0_attn_qact2_Frac  = 4.0;
    
    blocks_0_attn_qact2_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_qact2_Width, 'FractionLength', blocks_0_attn_qact2_Frac); 
    
    %% Attention  % proj
    
    % Getting parameters %
    blocks_0_attn_proj_weight = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_attn_proj_weight_fl_8.mat'); 
    blocks_0_attn_proj_bias = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_attn_proj_bias_fl_6.mat'); 
    blocks_0_attn_proj_weight = blocks_0_attn_proj_weight.data;
    blocks_0_attn_proj_bias = blocks_0_attn_proj_bias.data;
    
    % Fixed point number %
    blocks_0_attn_proj_weight_Width = 8;
    blocks_0_attn_proj_weight_Frac  = 8.0;
    blocks_0_attn_proj_bias_Width = 8;
    blocks_0_attn_proj_bias_Frac  = 6.0;
    
    blocks_0_attn_qact3_Width = 8;
    blocks_0_attn_qact3_Frac  = 5.0;
    
    blocks_0_attn_proj_weight_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_proj_weight_Width, 'FractionLength', blocks_0_attn_proj_weight_Frac); 
    blocks_0_attn_proj_bias_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_proj_bias_Width, 'FractionLength', blocks_0_attn_proj_bias_Frac); 
    blocks_0_attn_qact3_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_qact3_Width, 'FractionLength', blocks_0_attn_qact3_Frac); 
    
    %% Residual connection 1
    
    % Fixed point number %
    blocks_0_qact2_Width = 16;%8
    blocks_0_qact2_Frac  = 5.0;%3.0
    
    blocks_0_qact2_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_qact2_Width, 'FractionLength', blocks_0_qact2_Frac); 
    
    %% blocks_0_LayerNorm_2
    
    % Getting parameters %  
    % blocks_0_norm2_weight = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_norm2_weight_fl_6.mat'); 
    % blocks_0_norm2_bias = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_norm2_bias_fl_6.mat'); 

    blocks_0_norm2_weight = load('./mat_extract/deit_tiny/blocks/block_0/cpu_p_qudq_1_LN_weight_fl_5.mat'); 
    blocks_0_norm2_bias = load('./mat_extract/deit_tiny/blocks/block_0/cpu_p_qudq_1_LN_bias_fl_6.mat'); 

    blocks_0_norm2_weight = blocks_0_norm2_weight.data;
    blocks_0_norm2_bias = blocks_0_norm2_bias.data;
    
    % Fixed point number %
    blocks_0_qact3_Width = 8;
    blocks_0_qact3_Frac  = 3.0;
    
    blocks_0_qact3_Type{1}  = numerictype('Signed',1, 'WordLength', blocks_0_qact3_Width, 'FractionLength', blocks_0_qact3_Frac); 
    
    %% Mlp % fc1 & gelu
    
    % Getting parameters %  
    blocks_0_mlp_fc1_weight = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_mlp_fc1_weight_fl_8.mat'); 
    blocks_0_mlp_fc1_bias = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_mlp_fc1_bias_fl_6.mat'); 
    blocks_0_mlp_fc1_weight = blocks_0_mlp_fc1_weight.data;
    blocks_0_mlp_fc1_bias = blocks_0_mlp_fc1_bias.data;
  
    % Fixed point number %
    blocks_0_mlp_fc1_weight_Width = 8;
    blocks_0_mlp_fc1_weight_Frac  = 8.0;
    blocks_0_mlp_fc1_bias_Width = 8;
    blocks_0_mlp_fc1_bias_Frac  = 6.0;
    
    blocks_0_mlp_qact1_Width = 8;
    blocks_0_mlp_qact1_Frac  = 3.0;

    blocks_0_mlp_fc1_weight_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_mlp_fc1_weight_Width, 'FractionLength', blocks_0_mlp_fc1_weight_Frac); 
    blocks_0_mlp_fc1_bias_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_mlp_fc1_bias_Width, 'FractionLength', blocks_0_mlp_fc1_bias_Frac); 
    blocks_0_mlp_qact1_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_mlp_qact1_Width, 'FractionLength', blocks_0_mlp_qact1_Frac); 

    %% Mlp % fc2
    % Getting parameters % 
    blocks_0_mlp_fc2_weight = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_mlp_fc2_weight_fl_8.mat'); 
    blocks_0_mlp_fc2_bias = load('./mat_extract/deit_tiny/blocks/block_0/blocks_0_mlp_fc2_bias_fl_5.mat'); 
    blocks_0_mlp_fc2_weight = blocks_0_mlp_fc2_weight.data;
    blocks_0_mlp_fc2_bias = blocks_0_mlp_fc2_bias.data;
    
    % Fixed point number %
    blocks_0_mlp_fc2_weight_Width = 8;
    blocks_0_mlp_fc2_weight_Frac  = 8.0;
    blocks_0_mlp_fc2_bias_Width = 8;
    blocks_0_mlp_fc2_bias_Frac  = 5.0;
    
    blocks_0_mlp_qact2_Width = 8;
    blocks_0_mlp_qact2_Frac  = 4.0;
    
    blocks_0_mlp_fc2_weight_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_mlp_fc2_weight_Width, 'FractionLength', blocks_0_mlp_fc2_weight_Frac); 
    blocks_0_mlp_fc2_bias_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_mlp_fc2_bias_Width, 'FractionLength', blocks_0_mlp_fc2_bias_Frac); 
    blocks_0_mlp_qact2_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_mlp_qact2_Width, 'FractionLength', blocks_0_mlp_qact2_Frac); 
    
    %% Residual connection 2 
    
    % Fixed point number %
    blocks_0_qact4_Width = 16;%8
    blocks_0_qact4_Frac  = 5.0;%3.0
    
    blocks_0_qact4_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_qact4_Width, 'FractionLength', blocks_0_qact4_Frac); 
    
    %% INFERENCE %% INFERENCE %% INFERENCE %% INFERENCE %% INFERENCE %% INFERENCE
    
    %% blocks_0_LayerNorm_1
    % Note: LayerNorm is not quantized. Neither its weight nor bias was quantified.
    
    % 192*197
    qudq_qudq_x_And_qudq_pos_permute = permute(qudq_qudq_x_And_qudq_pos, [2, 1]);
    
    input_blocks_0_norm1 = single(qudq_qudq_x_And_qudq_pos_permute);
    
    epsilon_blocks_0_norm1 = 1e-6;
    dim_blocks_0_norm1 = 1; % Because the shape is (192,197), LayerNorm is performed on the first dimension
    scale_blocks_0_norm1 = blocks_0_norm1_weight;
    shift_blocks_0_norm1 = blocks_0_norm1_bias; 

    %----------------------------------------------------------------------
    %% phase_two
    if(Save==1)
        fid = fopen(strcat(MatOutPath,'tokens.data'), 'a');
        % layer_norm 
        % block0_norm1 
        fwrite(fid, blocks_0_norm1_weight, 'single');
        fwrite(fid, blocks_0_norm1_bias, 'single');
        fclose(fid);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    % result_blocks_0_norm1 = my_layernorm_repmat(input_blocks_0_norm1, epsilon_blocks_0_norm1, dim_blocks_0_norm1, scale_blocks_0_norm1, shift_blocks_0_norm1);
    
    result_blocks_0_norm1 = int_layernorm_repmat(input_blocks_0_norm1, epsilon_blocks_0_norm1, dim_blocks_0_norm1, scale_blocks_0_norm1, shift_blocks_0_norm1);

    result_blocks_0_norm1 = permute(result_blocks_0_norm1, [2, 1]);
    
    qudq_result_blocks_0_norm1=fi(result_blocks_0_norm1,blocks_0_qact1_Type{1}, MathType); %fif

    %% Attention % Linear to get qkv
    
    qudq_blocks_0_attn_qkv_weight = fi(blocks_0_attn_qkv_weight,blocks_0_attn_qkv_weight_Type{1}, MathType); %fif
    qudq_blocks_0_attn_qkv_bias = fi(blocks_0_attn_qkv_bias,blocks_0_attn_qkv_bias_Type{1}, MathType); %fif
    
    % blocks_0_attn_qkv = Linear(qudq_result_blocks_0_norm1, qudq_blocks_0_attn_qkv_weight, qudq_blocks_0_attn_qkv_bias);
    % qudq_blocks_0_attn_qkv = fi(blocks_0_attn_qkv,blocks_0_attn_qact1_Type{1}, MathType); %fif
    
    %----------------------------------------------------------------------
    %% phase_two
    % qudq_blocks_0_attn_qkv = fix_fc_deit(qudq_result_blocks_0_norm1, qudq_blocks_0_attn_qkv_weight, qudq_blocks_0_attn_qkv_bias, blocks_0_qact1_Type{1}, blocks_0_attn_qact1_Type{1}, MathType);
    qudq_result_blocks_0_norm1_reshape = reshape(qudq_result_blocks_0_norm1,1,197,192);
    qudq_blocks_0_attn_qkv_weight_reshape = reshape(qudq_blocks_0_attn_qkv_weight',1,1,192,576);
    qudq_blocks_0_attn_qkv_bias_permute = permute(qudq_blocks_0_attn_qkv_bias,[2,1]);
    qudq_blocks_0_attn_qkv = fix_conv(qudq_result_blocks_0_norm1_reshape,qudq_blocks_0_attn_qkv_weight_reshape,qudq_blocks_0_attn_qkv_bias_permute,1,1,0,1,blocks_0_qact1_Type{1}, blocks_0_attn_qact1_Type{1}, MathType);
    qudq_blocks_0_attn_qkv = reshape(qudq_blocks_0_attn_qkv,[],576);
    %test
    zeros_padded = zeros(1, 13, 192);
    zeros_padded=fi(zeros_padded,blocks_0_qact1_Type{1}, MathType);
    zeros_padded_Integer = storedInteger(zeros_padded);
    qudq_result_blocks_0_norm1_reshape_zero_padded = horzcat(qudq_result_blocks_0_norm1_reshape, zeros_padded_Integer);
    qudq_result_blocks_0_norm1_reshape_zero_padded_reshape = reshape(qudq_result_blocks_0_norm1_reshape_zero_padded,210,192);
    path = fullfile(MatOutPath, 'blocks_0/', 'qudq_result_blocks_0_norm1_reshape.data');
    fid = fopen(path, 'w');
    fwrite(fid, storedInteger(qudq_result_blocks_0_norm1_reshape_zero_padded_reshape), 'int8');
    fclose(fid);
    %end of test

    if(Save==1)
        if(OFFLINE_REORDER==0)%reorder weights on the ARM
            VEC_SIZE = 1; LANE_NUM = 1;
            fid = fopen(strcat(MatOutPath,'weights.data'), 'a');
        else
            fid = fopen(strcat(MatOutPath,sprintf('weights_vec%d_lane%d.data',VEC_SIZE,LANE_NUM)), 'a');
        end
        
        % % save weights and bias
        % block0 attn qkv
        fwrite(fid, storedInteger(vectorizeWeight(qudq_blocks_0_attn_qkv_weight_reshape, VEC_SIZE, LANE_NUM)), 'int8');
        fwrite(fid, storedInteger(vectorizeBias(qudq_blocks_0_attn_qkv_bias_permute, LANE_NUM)), 'int8');
        fclose(fid);
        
        % block0 LN 1
        disp 'Processing block0 LN 1  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_norm1.data'), 'w');
        fwrite(fid, storedInteger(qudq_result_blocks_0_norm1), 'int8');
        fclose(fid);

        % block0 attention qkv
        disp 'Processing block0 attention qkv  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_qkv.data'), 'w');
        fwrite(fid, storedInteger(qudq_blocks_0_attn_qkv), 'int8');
        fclose(fid);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %% Attention  % Pytorch reshape 
    % To achieve the same reshape effect in MATLAB as in pytorch, 
    % we need to perform the following three steps:
    % test = reshape(qudq_blocks_0_attn_qkv, 197,3,3,64);%phase_two
    % x_cpu_numpy = load('./x_cpu_numpy.mat');
    % x_cpu_numpy_reshape = reshape(x_cpu_numpy.data,[],576);
    
    qkv_permute = permute(qudq_blocks_0_attn_qkv, [2,1]);
    qkv_permute_reshape = reshape(qkv_permute, 64,3,3,197);
    qkv_permute_reshape_permute = permute(qkv_permute_reshape, [4,3,2,1]); 
    
    %% Attention  % Pytorch permute
    
    % Pytorch permute
    qkv_permute_reshape_permute_permute = permute(qkv_permute_reshape_permute, [2,3,1,4]);

    %% Attention  % to get q, k, v
    
    % Split the three elements of the first dimension
    attn_q = squeeze(qkv_permute_reshape_permute_permute(1, :, :, :));
    attn_k = squeeze(qkv_permute_reshape_permute_permute(2, :, :, :));
    attn_v = squeeze(qkv_permute_reshape_permute_permute(3, :, :, :));
    
    %% Attention  % SDP
    
    head_dim = 64;
    scale = power(head_dim, -0.5);
    
    attn_head_1_q = squeeze(attn_q(1, :, :));
    attn_head_1_k = squeeze(attn_k(1, :, :));
    attn_head_1_v = squeeze(attn_v(1, :, :));
    
    attn_head_2_q = squeeze(attn_q(2, :, :));
    attn_head_2_k = squeeze(attn_k(2, :, :));
    attn_head_2_v = squeeze(attn_v(2, :, :));
    
    attn_head_3_q = squeeze(attn_q(3, :, :));
    attn_head_3_k = squeeze(attn_k(3, :, :));
    attn_head_3_v = squeeze(attn_v(3, :, :));
    
    % SDP
    attn_head_1 = (single(attn_head_1_q) * single(permute(attn_head_1_k, [ 2, 1]))) * scale;

    %----------------------------------------------------------------------
    %% phase_two
    % Fixed point number %
    blocks_0_attn_qact_attn1_Width_scale = 8;
    blocks_0_attn_qact_attn1_Frac_scale  = blocks_0_attn_qact_attn1_Frac-3.0;%3.0-3.0
    blocks_0_attn_qact_attn1_Type_scale{1} = numerictype('Signed',1, 'WordLength', blocks_0_attn_qact_attn1_Width_scale, 'FractionLength', blocks_0_attn_qact_attn1_Frac_scale);
    
    attn_head_1_q_reshape = reshape(attn_head_1_q,1,197,64);%phase_two
    attn_head_1_k_permute_reshape = reshape(attn_head_1_k',1,1,64,197); 
    attn_qk_bias = fi(zeros(197, 1),blocks_0_attn_qact1_Type{1},MathType);
    attn_head_1_conv = fix_conv(attn_head_1_q_reshape,attn_head_1_k_permute_reshape,attn_qk_bias,1,1,0,1,blocks_0_attn_qact1_Type{1}, blocks_0_attn_qact_attn1_Type_scale{1}, MathType);
    attn_head_1_conv = reshape(attn_head_1_conv,[],197);
    attn_head_1_ref = fi(attn_head_1,blocks_0_attn_qact_attn1_Type{1}, MathType);
    if(Save==1)
        % block0 attention matrix multiplication 1
        disp 'Processing block0 attention matrix multiplication 1  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_head_1.data'), 'w');
        fwrite(fid, storedInteger(attn_head_1_conv), 'int8');
        fclose(fid);
    end
    attn_head_1_conv_Integer = storedInteger(attn_head_1_conv);
    attn_head_1_ref_Integer = storedInteger(attn_head_1_ref);
    if(IsCheck==1)
        fix_check_2dim(attn_head_1_conv_Integer,attn_head_1_ref_Integer);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %%
    attn_head_1 = single(attn_head_1);
    attn_head_2 = (single(attn_head_2_q) * single(permute(attn_head_2_k, [ 2, 1]))) * scale;

    %----------------------------------------------------------------------
    %% phase_two
    attn_head_2_q_reshape = reshape(attn_head_2_q,1,197,64);
    attn_head_2_k_permute_reshape = reshape(attn_head_2_k',1,1,64,197);
    attn_head_2_conv = fix_conv(attn_head_2_q_reshape,attn_head_2_k_permute_reshape,attn_qk_bias,1,1,0,1,blocks_0_attn_qact1_Type{1}, blocks_0_attn_qact_attn1_Type_scale{1}, MathType);
    attn_head_2_conv = reshape(attn_head_2_conv,[],197);
    attn_head_2_ref = fi(attn_head_2,blocks_0_attn_qact_attn1_Type{1}, MathType);
    if(Save==1)
        % block0 attention matrix multiplication 2
        disp 'Processing block0 attention matrix multiplication 2  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_head_2.data'), 'w');
        fwrite(fid, storedInteger(attn_head_2_conv), 'int8');
        fclose(fid);
    end
    attn_head_2_conv_Integer = storedInteger(attn_head_2_conv);
    attn_head_2_ref_Integer = storedInteger(attn_head_2_ref);
    if(IsCheck==1)
        fix_check_2dim(attn_head_2_conv_Integer,attn_head_2_ref_Integer);
    end
    %end of phase_two
    %----------------------------------------------------------------------

    %%
    attn_head_2 = single(attn_head_2);
    attn_head_3 = (single(attn_head_3_q) * single(permute(attn_head_3_k, [ 2, 1]))) * scale;

    %----------------------------------------------------------------------
    %% phase_two
    attn_head_3_q_reshape = reshape(attn_head_3_q,1,197,64);
    attn_head_3_k_permute_reshape = reshape(attn_head_3_k',1,1,64,197);
    attn_head_3_conv = fix_conv(attn_head_3_q_reshape,attn_head_3_k_permute_reshape,attn_qk_bias,1,1,0,1,blocks_0_attn_qact1_Type{1}, blocks_0_attn_qact_attn1_Type_scale{1}, MathType);
    attn_head_3_conv = reshape(attn_head_3_conv,[],197);
    attn_head_3_ref = fi(attn_head_3,blocks_0_attn_qact_attn1_Type{1}, MathType);
    if(Save==1)
        % block0 attention matrix multiplication 3
        disp 'Processing block0 attention matrix multiplication 3  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_head_3.data'), 'w');
        fwrite(fid, storedInteger(attn_head_3_conv), 'int8');
        fclose(fid);
    end
    attn_head_3_conv_Integer = storedInteger(attn_head_3_conv);
    attn_head_3_ref_Integer = storedInteger(attn_head_3_ref);
    if(IsCheck==1)
       fix_check_2dim(attn_head_3_conv_Integer,attn_head_3_ref_Integer);
    end
    %end of phase_two
    %----------------------------------------------------------------------

    %%
    attn_head_3 = single(attn_head_3);
    
    attn = cat(1, attn_head_1, attn_head_2, attn_head_3); % qudq_of_patch_embed_reshape_3 fif

    attn = permute(attn, [2,1]); 
    attn = reshape(attn, 197,[],3);
    attn = permute(attn, [3,2,1]); 
    
    blocks_0_attn_qact_attn1 = fi(attn,blocks_0_attn_qact_attn1_Type{1}, MathType); %fif

    %----------------------------------------------------------------------
    %% phase_two
    blocks_0_attn_qact_attn1_vertcat =vertcat(attn_head_1_conv, attn_head_2_conv, attn_head_3_conv);
    blocks_0_attn_qact_attn1_vertcat = permute(blocks_0_attn_qact_attn1_vertcat, [2,1]); 
    blocks_0_attn_qact_attn1_vertcat = reshape(blocks_0_attn_qact_attn1_vertcat, 197,[],3);
    blocks_0_attn_qact_attn1_vertcat = permute(blocks_0_attn_qact_attn1_vertcat, [3,2,1]); 
    blocks_0_attn_qact_attn1_vertcat_Integer = storedInteger(blocks_0_attn_qact_attn1_vertcat);
    blocks_0_attn_qact_attn1_Integer = storedInteger(blocks_0_attn_qact_attn1);
    if(IsCheck==1)
        fix_check_3dim(blocks_0_attn_qact_attn1_Integer,blocks_0_attn_qact_attn1_vertcat_Integer);
    end
    %end of phase_two
    %----------------------------------------------------------------------
    
    %% Attention  % softmax
    
    % x = single(blocks_0_attn_qact_attn1);
    %blocks_0_attn_log_int_softmax = ViTSoftmax(single(blocks_0_attn_qact_attn1), 3);
    blocks_0_attn_log_int_softmax = ViTSoftmax_LUT(single(blocks_0_attn_qact_attn1), 3, MathType);
    
    qudq_blocks_0_attn_log_int_softmax = fi(blocks_0_attn_log_int_softmax,blocks_0_attn_log_int_softmax_Type{1}, MathType); %fif
    
    %% Attention  % attn @ v
    
    qudq_blocks_0_attn_log_int_softmax_head_1 = squeeze(qudq_blocks_0_attn_log_int_softmax(1, :, :));
    
    qudq_blocks_0_attn_log_int_softmax_head_2 = squeeze(qudq_blocks_0_attn_log_int_softmax(2, :, :));
    
    qudq_blocks_0_attn_log_int_softmax_head_3 = squeeze(qudq_blocks_0_attn_log_int_softmax(3, :, :));
    %phase_two
    qudq_blocks_0_attn_log_int_softmax_head_1_Intger = storedInteger(qudq_blocks_0_attn_log_int_softmax_head_1);
    qudq_blocks_0_attn_log_int_softmax_head_2_Intger = storedInteger(qudq_blocks_0_attn_log_int_softmax_head_2);
    qudq_blocks_0_attn_log_int_softmax_head_3_Intger = storedInteger(qudq_blocks_0_attn_log_int_softmax_head_3);
    b_double = qudq_blocks_0_attn_log_int_softmax_head_3_Intger;
    fid = fopen(strcat(MatOutPath,'qudq_blocks_0_attn_log_int_softmax_head_3.data'), 'w');
    %fwrite(fid, storedInteger(blocks_0_attn_qact_attn1), 'single');
 
    fwrite(fid, b_double, 'int8');
    fclose(fid);
    %disp(numel(b_double));
    disp(b_double(1:10));
    %end of phase_two
    
    blocks_0_attn_head_1_output = permute((single(qudq_blocks_0_attn_log_int_softmax_head_1) * single(attn_head_1_v)), [2, 1]);

    %----------------------------------------------------------------------
    %% phase_two
    qudq_blocks_0_attn_log_int_softmax_head_1_reshape = reshape(qudq_blocks_0_attn_log_int_softmax_head_1,1,197,197);%phase_two
    attn_head_1_v_reshape = reshape(attn_head_1_v,1,1,197,64);
    qk_softmax_bias = fi(zeros(64, 1),blocks_0_attn_qact1_Type{1},MathType);
    blocks_0_attn_head_1_output_conv = fix_conv(qudq_blocks_0_attn_log_int_softmax_head_1_reshape,attn_head_1_v_reshape,qk_softmax_bias,1,1,0,1,blocks_0_attn_log_int_softmax_Type{1}, blocks_0_attn_qact2_Type{1}, MathType);
    if(Save==1)
        % block0 attention matrix multiplication 4
        disp 'Processing block0 attention matrix multiplication 4  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_head_1_output.data'), 'w');
        fwrite(fid, storedInteger(blocks_0_attn_head_1_output_conv), 'int8');
        fclose(fid);
    end
    blocks_0_attn_head_1_output_conv = permute(reshape(blocks_0_attn_head_1_output_conv,[],64),[2,1]);
    blocks_0_attn_head_1_output_ref = fi(blocks_0_attn_head_1_output,blocks_0_attn_qact2_Type{1}, MathType);
    if(IsCheck==1)
       fix_check_2dim(blocks_0_attn_head_1_output_ref,blocks_0_attn_head_1_output_conv);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %%
    blocks_0_attn_head_2_output = permute((single(qudq_blocks_0_attn_log_int_softmax_head_2) * single(attn_head_2_v)), [2, 1]);
    
    %----------------------------------------------------------------------
    %% phase_two
    qudq_blocks_0_attn_log_int_softmax_head_2_reshape = reshape(qudq_blocks_0_attn_log_int_softmax_head_2,1,197,197);
    attn_head_2_v_reshape = reshape(attn_head_2_v,1,1,197,64);
    qk_softmax_bias = fi(zeros(64, 1),blocks_0_attn_qact1_Type{1},MathType);
    blocks_0_attn_head_2_output_conv = fix_conv(qudq_blocks_0_attn_log_int_softmax_head_2_reshape,attn_head_2_v_reshape,qk_softmax_bias,1,1,0,1,blocks_0_attn_log_int_softmax_Type{1}, blocks_0_attn_qact2_Type{1}, MathType);
    if(Save==1)
        % block0 attention matrix multiplication 5
        disp 'Processing block0 attention matrix multiplication 5  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_head_2_output.data'), 'w');
        fwrite(fid, storedInteger(blocks_0_attn_head_2_output_conv), 'int8');
        fclose(fid);
    end
    blocks_0_attn_head_2_output_conv = permute(reshape(blocks_0_attn_head_2_output_conv,[],64),[2,1]);
    blocks_0_attn_head_2_output_ref = fi(blocks_0_attn_head_2_output,blocks_0_attn_qact2_Type{1}, MathType);
    if(IsCheck==1)
       fix_check_2dim(blocks_0_attn_head_2_output_ref,blocks_0_attn_head_2_output_conv);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %%
    blocks_0_attn_head_3_output = permute((single(qudq_blocks_0_attn_log_int_softmax_head_3) * single(attn_head_3_v)), [2, 1]);
    
    %----------------------------------------------------------------------
    %% phase_two
    qudq_blocks_0_attn_log_int_softmax_head_3_reshape = reshape(qudq_blocks_0_attn_log_int_softmax_head_3,1,197,197);
    attn_head_3_v_reshape = reshape(attn_head_3_v,1,1,197,64);
    qk_softmax_bias = fi(zeros(64, 1),blocks_0_attn_qact1_Type{1},MathType);
    blocks_0_attn_head_3_output_conv = fix_conv(qudq_blocks_0_attn_log_int_softmax_head_3_reshape,attn_head_3_v_reshape,qk_softmax_bias,1,1,0,1,blocks_0_attn_log_int_softmax_Type{1}, blocks_0_attn_qact2_Type{1}, MathType);
    if(Save==1)
        % block0 attention matrix multiplication 6
        disp 'Processing block0 attention matrix multiplication 6  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_head_3_output.data'), 'w');
        fwrite(fid, storedInteger(blocks_0_attn_head_3_output_conv), 'int8');
        fclose(fid);
    end
    blocks_0_attn_head_3_output_conv = permute(reshape(blocks_0_attn_head_3_output_conv,[],64),[2,1]);
    blocks_0_attn_head_3_output_ref = fi(blocks_0_attn_head_3_output,blocks_0_attn_qact2_Type{1}, MathType);
    if(IsCheck==1)
       fix_check_2dim(blocks_0_attn_head_3_output_ref,blocks_0_attn_head_3_output_conv);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %%
    blocks_0_attn_head_output = cat(1, single(blocks_0_attn_head_1_output), single(blocks_0_attn_head_2_output), single(blocks_0_attn_head_3_output)); % qudq_of_patch_embed_reshape_3 fif
    
    blocks_0_attn_head_output = permute(blocks_0_attn_head_output, [2,1]); 
    
    qudq_blocks_0_attn_head_output = fi(blocks_0_attn_head_output,blocks_0_attn_qact2_Type{1}, MathType);

    %----------------------------------------------------------------------
    %% phase_two
    blocks_0_attn_head_output_vertcat = vertcat(blocks_0_attn_head_1_output_conv,blocks_0_attn_head_2_output_conv,blocks_0_attn_head_3_output_conv);
    
    blocks_0_attn_head_output_vertcat = permute(blocks_0_attn_head_output_vertcat,[2,1]);
    
    if(IsCheck==1)
       fix_check_2dim(qudq_blocks_0_attn_head_output,blocks_0_attn_head_output_vertcat);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %% Attention  % proj
    
    qudq_blocks_0_attn_proj_weight = fi(blocks_0_attn_proj_weight,blocks_0_attn_proj_weight_Type{1}, MathType); %fif
    qudq_blocks_0_attn_proj_bias = fi(blocks_0_attn_proj_bias,blocks_0_attn_proj_bias_Type{1}, MathType); %fif
    
    
    % blocks_0_attn_proj = Linear(qudq_blocks_0_attn_head_output, qudq_blocks_0_attn_proj_weight, qudq_blocks_0_attn_proj_bias);
    % 
    % qudq_blocks_0_attn_proj = fi(blocks_0_attn_proj,blocks_0_attn_qact3_Type{1}, MathType); 

    %----------------------------------------------------------------------
    %% phase_two
    % qudq_blocks_0_attn_proj = fix_fc_deit(qudq_blocks_0_attn_head_output, qudq_blocks_0_attn_proj_weight, qudq_blocks_0_attn_proj_bias, blocks_0_attn_qact2_Type{1}, blocks_0_attn_qact3_Type{1}, MathType);
    qudq_blocks_0_attn_head_output_reshape = reshape(qudq_blocks_0_attn_head_output,1,197,192);
    qudq_blocks_0_attn_proj_weight_reshape = reshape(qudq_blocks_0_attn_proj_weight',1,1,192,192);
    qudq_blocks_0_attn_proj_bias_permute = permute(qudq_blocks_0_attn_proj_bias,[2,1]);
    qudq_blocks_0_attn_proj = fix_conv(qudq_blocks_0_attn_head_output_reshape,qudq_blocks_0_attn_proj_weight_reshape,qudq_blocks_0_attn_proj_bias_permute,1,1,0,1,blocks_0_attn_qact2_Type{1}, blocks_0_attn_qact3_Type{1}, MathType);
    qudq_blocks_0_attn_proj = reshape(qudq_blocks_0_attn_proj,[],192);
    if(Save==1)
        if(OFFLINE_REORDER==0)%reorder weights on the ARM
            VEC_SIZE = 1; LANE_NUM = 1;
            fid = fopen(strcat(MatOutPath,'weights.data'), 'a');
        else
            fid = fopen(strcat(MatOutPath,sprintf('weights_vec%d_lane%d.data',VEC_SIZE,LANE_NUM)), 'a');
        end

        % % save weights and bias
        % block0 attn proj
        fwrite(fid, storedInteger(vectorizeWeight(qudq_blocks_0_attn_proj_weight_reshape, VEC_SIZE, LANE_NUM)), 'int8');
        fwrite(fid, storedInteger(vectorizeBias(qudq_blocks_0_attn_proj_bias_permute, LANE_NUM)), 'int8');
        fclose(fid);

        % block0 attention proj
        disp 'Processing block0 attention proj  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_attn_proj.data'), 'w');
        fwrite(fid, storedInteger(qudq_blocks_0_attn_proj), 'int8');
        fclose(fid);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %% Residual connection 1
    
    % In this case, the int8 fixed-point number is converted to a single before the calculation is performed; 
    % note how this is implemented in hardware.
    blocks_0_x_1 = single(qudq_qudq_x_And_qudq_pos) + single(qudq_blocks_0_attn_proj);

    qudq_blocks_0_x_1 = fi(blocks_0_x_1,blocks_0_qact2_Type{1}, MathType); %fif
    qudq_blocks_0_x_1_Integer = storedInteger(qudq_blocks_0_x_1);
    %% blocks_0_LayerNorm_2
    
    qudq_blocks_0_x_1_permute = permute(qudq_blocks_0_x_1, [2, 1]);
    
    input_blocks_0_norm2 = single(qudq_blocks_0_x_1_permute);
    epsilon_blocks_0_norm2 = 1e-6;
    dim_blocks_0_norm2 = 1; 
    scale_blocks_0_norm2 = blocks_0_norm2_weight; 
    shift_blocks_0_norm2 = blocks_0_norm2_bias;

    %----------------------------------------------------------------------
    %% phase_two
    if(Save==1)
        fid = fopen(strcat(MatOutPath,'tokens.data'), 'a');
        % layer_norm 
        % block0_norm2 
        fwrite(fid, blocks_0_norm2_weight, 'single');
        fwrite(fid, blocks_0_norm2_bias, 'single');
        fclose(fid);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    % result_blocks_0_norm2 = my_layernorm_repmat(input_blocks_0_norm2, epsilon_blocks_0_norm2, dim_blocks_0_norm2, scale_blocks_0_norm2, shift_blocks_0_norm2);
   
    result_blocks_0_norm2 = int_layernorm_repmat(input_blocks_0_norm2, epsilon_blocks_0_norm2, dim_blocks_0_norm2, scale_blocks_0_norm2, shift_blocks_0_norm2);

    result_blocks_0_norm2 = permute(result_blocks_0_norm2, [2, 1]);
    
    qudq_result_blocks_0_norm2=fi(result_blocks_0_norm2,blocks_0_qact3_Type{1}, MathType); %fif

    %% Mlp % fc1 & gelu
    
    qudq_blocks_0_mlp_fc1_weight = fi(blocks_0_mlp_fc1_weight,blocks_0_mlp_fc1_weight_Type{1}, MathType); %fif
    qudq_blocks_0_mlp_fc1_bias = fi(blocks_0_mlp_fc1_bias,blocks_0_mlp_fc1_bias_Type{1}, MathType); %fif
    
    blocks_0_mlp_fc1 = Linear(qudq_result_blocks_0_norm2, qudq_blocks_0_mlp_fc1_weight, qudq_blocks_0_mlp_fc1_bias);

    %----------------------------------------------------------------------
    %% phase_two
    % blocks_0_mlp_qact0_t_Width = 16;
    % blocks_0_mlp_qact0_t_Frac = 11.0;
    % blocks_0_mlp_qact0_t_Type{1} = numerictype('Signed',1, 'WordLength', blocks_0_mlp_qact0_t_Width, 'FractionLength', blocks_0_mlp_qact0_t_Frac); 
    % blocks_0_mlp_fc1_fi = fi(blocks_0_mlp_fc1,blocks_0_mlp_qact0_t_Type{1}, MathType); %fif
    % % blocks_0_mlp_fc1_fi = single(blocks_0_mlp_fc1_fi);
    % blocks_0_mlp_fc1_gelu_fi = ViT_gelu(blocks_0_mlp_fc1_fi);
    % 
    % blocks_0_mlp_fc1_gelu_fi_gelu = fi(blocks_0_mlp_fc1_gelu_fi,blocks_0_mlp_qact1_Type{1}, MathType); %fif
    
    % end of phase_two
    %----------------------------------------------------------------------

    %% 
    % Perform the GELU operation
    blocks_0_mlp_fc1 = single(blocks_0_mlp_fc1);
    blocks_0_mlp_fc1_gelu = ViT_gelu(blocks_0_mlp_fc1);

    qudq_blocks_0_mlp_fc1_gelu = fi(blocks_0_mlp_fc1_gelu,blocks_0_mlp_qact1_Type{1}, MathType); %fif
    
    %----------------------------------------------------------------------
    %% phase_two
    qudq_result_blocks_0_norm2_reshape = reshape(qudq_result_blocks_0_norm2,1,197,192);
    qudq_blocks_0_mlp_fc1_weight_reshape = reshape(qudq_blocks_0_mlp_fc1_weight',1,1,192,768);
    qudq_blocks_0_mlp_fc1_bias_permute = permute(qudq_blocks_0_mlp_fc1_bias,[2,1]);
    qudq_blocks_0_mlp_fc1_gelu_conv = fix_conv_gelu(qudq_result_blocks_0_norm2_reshape,qudq_blocks_0_mlp_fc1_weight_reshape,qudq_blocks_0_mlp_fc1_bias_permute,1,1,0,1,blocks_0_qact3_Type{1}, blocks_0_mlp_qact1_Type{1}, MathType);
    qudq_blocks_0_mlp_fc1_gelu_conv = reshape(qudq_blocks_0_mlp_fc1_gelu_conv,[],768);
    %test
    zeros_padded = zeros(1, 13, 192);
    zeros_padded=fi(zeros_padded,blocks_0_qact3_Type{1}, MathType);
    zeros_padded_Integer = storedInteger(zeros_padded);
    qudq_result_blocks_0_norm2_reshape_zero_padded = horzcat(qudq_result_blocks_0_norm2_reshape, zeros_padded_Integer);
    qudq_blocks_0_mlp_fc1_gelu_conv_Int_reshape_zero_padded_reshape = reshape(qudq_result_blocks_0_norm2_reshape_zero_padded,210,192);
    path = fullfile(MatOutPath, 'blocks_0/', 'qudq_result_blocks_0_norm2_reshape.data');
    fid = fopen(path, 'w');
    fwrite(fid, storedInteger(qudq_blocks_0_mlp_fc1_gelu_conv_Int_reshape_zero_padded_reshape), 'int8');
    fclose(fid);
    %end of test

    if(Save==1)
        if(OFFLINE_REORDER==0)%reorder weights on the ARM
            VEC_SIZE = 1; LANE_NUM = 1;
            fid = fopen(strcat(MatOutPath,'weights.data'), 'a');
        else
            fid = fopen(strcat(MatOutPath,sprintf('weights_vec%d_lane%d.data',VEC_SIZE,LANE_NUM)), 'a');
        end

        % % save weights and bias
        % patch_embed.proj
        fwrite(fid, storedInteger(vectorizeWeight(qudq_blocks_0_mlp_fc1_weight_reshape, VEC_SIZE, LANE_NUM)), 'int8');
        fwrite(fid, storedInteger(vectorizeBias(qudq_blocks_0_mlp_fc1_bias_permute, LANE_NUM)), 'int8');
        fclose(fid);

        % block0 block0 mlp fc1
        disp 'Processing block0 mlp fc1  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_mlp_fc1_gelu.data'), 'w');
        fwrite(fid, storedInteger(qudq_blocks_0_mlp_fc1_gelu_conv), 'int8');
        fclose(fid);
    end
    if(IsCheck==1)
       fix_check_2dim(qudq_blocks_0_mlp_fc1_gelu,qudq_blocks_0_mlp_fc1_gelu_conv);
    end
    % end of phase_two
    %----------------------------------------------------------------------
    
    %% Mlp % fc2
    
    qudq_blocks_0_mlp_fc1_weight = fi(blocks_0_mlp_fc2_weight,blocks_0_mlp_fc2_weight_Type{1}, MathType); %fif
    qudq_blocks_0_mlp_fc1_bias = fi(blocks_0_mlp_fc2_bias,blocks_0_mlp_fc2_bias_Type{1}, MathType); %fif
    
    % Perform Linear operations
    % blocks_0_mlp_fc2 = Linear(qudq_blocks_0_mlp_fc1_gelu, qudq_blocks_0_mlp_fc1_weight, qudq_blocks_0_mlp_fc1_bias);
    % 
    % qudq_blocks_0_mlp_fc2 = fi(blocks_0_mlp_fc2,blocks_0_mlp_qact2_Type{1}, MathType); %fif

    %----------------------------------------------------------------------
    %% phase_two
    qudq_blocks_0_mlp_fc2 = fix_fc_deit(qudq_blocks_0_mlp_fc1_gelu_conv, qudq_blocks_0_mlp_fc1_weight, qudq_blocks_0_mlp_fc1_bias, blocks_0_mlp_qact1_Type{1}, blocks_0_mlp_qact2_Type{1}, MathType);
    
    qudq_blocks_0_mlp_fc1_gelu_reshape = reshape(qudq_blocks_0_mlp_fc1_gelu_conv,1,197,768);
    qudq_blocks_0_mlp_fc1_weight_reshape = reshape(qudq_blocks_0_mlp_fc1_weight',1,1,768,192);
    qudq_blocks_0_mlp_fc1_bias_permute = permute(qudq_blocks_0_mlp_fc1_bias,[2,1]);
    qudq_blocks_0_mlp_fc2_conv = fix_conv(qudq_blocks_0_mlp_fc1_gelu_reshape,qudq_blocks_0_mlp_fc1_weight_reshape,qudq_blocks_0_mlp_fc1_bias_permute,1,1,0,1,blocks_0_mlp_qact1_Type{1}, blocks_0_mlp_qact2_Type{1}, MathType);
    qudq_blocks_0_mlp_fc2_conv = reshape(qudq_blocks_0_mlp_fc2_conv,[],192);
    
    if(Save==1)
        if(OFFLINE_REORDER==0)%reorder weights on the ARM
            VEC_SIZE = 1; LANE_NUM = 1;
            fid = fopen(strcat(MatOutPath,'weights.data'), 'a');
        else
            fid = fopen(strcat(MatOutPath,sprintf('weights_vec%d_lane%d.data',VEC_SIZE,LANE_NUM)), 'a');
        end

        % % save weights and bias
        % block0 mlp fc2
        fwrite(fid, storedInteger(vectorizeWeight(qudq_blocks_0_mlp_fc1_weight_reshape, VEC_SIZE, LANE_NUM)), 'int8');
        fwrite(fid, storedInteger(vectorizeBias(qudq_blocks_0_mlp_fc1_bias_permute, LANE_NUM)), 'int8');
        fclose(fid);

        % block0 mlp fc2
        disp 'Processing block0 mlp fc2  ...'
        fid = fopen(fullfile(MatOutPath, folderName, 'blocks_0_mlp_fc2.data'), 'w');
        fwrite(fid, storedInteger(qudq_blocks_0_mlp_fc2_conv), 'int8');
        fclose(fid);
    end
    if(IsCheck==1)
       fix_check_2dim(qudq_blocks_0_mlp_fc2,qudq_blocks_0_mlp_fc2_conv);
    end
    % end of phase_two
    %----------------------------------------------------------------------

    %% Residual connection 2 
    
    % In this case, the int8 fixed-point number is converted to a single before the calculation is performed; 
    % note how this is implemented in hardware.

    blocks_0_x_2 = single(qudq_blocks_0_x_1) + single(qudq_blocks_0_mlp_fc2);
    
    qudq_blocks_0_x_2 = fi(blocks_0_x_2,blocks_0_qact4_Type{1}, MathType); %fif
    qudq_blocks_0_x_1_Integer = storedInteger(qudq_blocks_0_x_1);
    qudq_blocks_0_x_2_Integer = storedInteger(qudq_blocks_0_x_2);
%     % Check
%     p_qudq_blocks_0_x_2 = load('./mat_extract/deit_tiny/blocks/p_qudq_blocks_0_inner_x_2.mat'); 
%     p_qudq_blocks_0_x_2 = reshape(p_qudq_blocks_0_x_2.data,size(p_qudq_blocks_0_x_2.data,2),size(p_qudq_blocks_0_x_2.data,3));
%      
%     disp('p_qudq_blocks_0_x_2');
%     fix_check(p_qudq_blocks_0_x_2,qudq_blocks_0_x_2);
    
    result = qudq_blocks_0_x_2;
    
end
