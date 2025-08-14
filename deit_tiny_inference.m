
% Forward path implementation of deit_tiny fixedpoint

dbstop if error
clc;
clear;
restoredefaultpath();
path(path,'./src');
path(path,'./src/common');
path(path,'./src/core');
path(path,'./src/vit');

%%
global Save;  Save = 1;
global OFFLINE_REORDER; OFFLINE_REORDER = 0; 
% global VEC_SIZE; VEC_SIZE = 8; 
% global LANE_NUM; LANE_NUM = 16;
% global ALIGN_SIZE;ALIGN_SIZE = 16;
global VEC_SIZE; VEC_SIZE = 4; 
global LANE_NUM; LANE_NUM = 8;
global ALIGN_SIZE;ALIGN_SIZE = 8;
global IsCheck;  IsCheck = 0;

global MatOutPath; MatOutPath = "./mat_output/Deit_Tiny/";

%% MathType

MathType   = fimath('RoundingMethod', 'Nearest', 'OverflowAction', 'Saturate', 'ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');

%%

disp 'Running Deit_Tiny in forward path ...' 
BatchSize = 1;
Iteration = 1; 
ImageNum = BatchSize * Iteration; % total number of image in the database, used to generate random image index

% ImageIndexPrefix = '../data_set/ILSVRC2012_img_val/ILSVRC2012_val_';
% LabelPath= '../data_set/val.txt';

% ImageIndexPrefix = 'K:/HCProjects/data_set/ILSVRC2012_img_val/ILSVRC2012_val_'; 
% LabelPath= 'K:/HCProjects/data_set/val.txt';

ImageIndexPrefix = 'D:/data_set/ILSVRC2012_img_val/ILSVRC2012_val_'; 
LabelPath= './data_set/val.txt';

% ImageIndexPrefix = '/home/ice/Disk02/ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_';
% LabelPath= '/home/ice/Disk02/ImageNet/val.txt';

accuracy = zeros(1,2);  % top_1, top_5

%Loging file
% LogName = './log.txt';
% LogFid  = fopen(LogName,'w');
LogName = './log.txt';
LogFid  = fopen(LogName,'a+');

% Record the program execution time
log_file = 'run_program_time_log.txt';
log_fid = fopen(log_file, 'a+');

start_time = datestr(now, 'yyyy-mm-dd HH:MM:SS'); % Get the current time as a string
fprintf(log_fid, '\n%s: Program started\n', start_time);

%% Prepare input image.

disp 'Preparing image data ...' 
ImageIndex = (1:1:ImageNum)'; % original
FirstImgIdx = 0;

fid_label = fopen(LabelPath, 'r');
data_validation = textscan(fid_label, '%s %f');
fclose(fid_label);

ImageName = data_validation{1};
LabelTable = data_validation{2};

%% Start processing
tic;
disp 'Start processing ...'

for Iter=0:Iteration-1

ImageIndexBatch = FirstImgIdx+ImageIndex(1+Iter*BatchSize:BatchSize+Iter*BatchSize);  % Image index processed by each batch
data = prepareImage_DeiT_Tiny(ImageIndexPrefix, ImageIndexBatch, 224); % pic are loaded from jpeg

data = permute(data ,[4, 3, 2, 1]); % N-C-H-W
label = getLabel(LabelTable, ImageIndexBatch);

for Batch=1:BatchSize
fprintf('Processing Iteration = %d, Batch = %d\n', Iter, Batch);

%% Patch_embedding

% tokens
weight_proj = load('./mat_extract/deit_tiny/patch_embed_proj_weight_9.mat');
bias_proj = load('./mat_extract/deit_tiny/patch_embed_proj_bias_6.mat');

InputInWidth = 8;
InputInFrac  = 6.0;

OutputWidth  = 8;
OutputFrac   = 3.0;

Width_Weight = 8;
Frac_Weight  = 9.0;

Width_Bias   = 8;
Frac_Bias    = 6.0;
                                     
InputInType{1}  = numerictype('Signed',1, 'WordLength', InputInWidth, 'FractionLength', InputInFrac); 
OutputType{1}  = numerictype('Signed',1, 'WordLength', OutputWidth, 'FractionLength', OutputFrac); 
WeightType{1}  = numerictype('Signed',1, 'WordLength', Width_Weight, 'FractionLength', Frac_Weight); 
BiasType{1}  = numerictype('Signed',1, 'WordLength', Width_Bias, 'FractionLength', Frac_Bias); 

qudq_Input=fi(data(Batch,:,:,:), InputInType{1}, MathType);
qudq_weight = fi(weight_proj.data, WeightType{1}, MathType);
qudq_bias = fi(bias_proj.data, BiasType{1}, MathType);
qudq_bias = permute(qudq_bias,[2,1]);

qudq_Input_3d = reshape(qudq_Input,size(qudq_Input,2),size(qudq_Input,3),size(qudq_Input,4));%C-H-W
qudq_Input_3d_permute = permute(qudq_Input_3d, [3, 2, 1]); %W-H-C
qudq_weight_permute = permute(qudq_weight, [4, 3, 2, 1]); %W-H-C-M

if(Save==1)
    if(OFFLINE_REORDER==0)%reorder weights on the ARM
        VEC_SIZE = 1; LANE_NUM = 1;
        fid = fopen(strcat(MatOutPath,'weights.data'), 'w');
    else
        fid = fopen(strcat(MatOutPath,sprintf('weights_vec%d_lane%d.data',VEC_SIZE,LANE_NUM)), 'w');
    end
    
    % % save weights and bias
    % patch_embed.proj
    fwrite(fid, storedInteger(vectorizeWeight(qudq_weight_permute, VEC_SIZE, LANE_NUM)), 'int8');
    fwrite(fid, storedInteger(vectorizeBias(qudq_bias, LANE_NUM)), 'int8');
    fclose(fid);
end

conv0_out = fix_conv(qudq_Input_3d_permute, qudq_weight_permute, qudq_bias, 16 , 16 , 0 , 1, InputInType{1} , OutputType{1}, MathType); 
conv0_out_reshape_1 = reshape(conv0_out, [], 192);

% cls_token
p_cls_token_expand_B_neg1_neg1 = load('./mat_extract/deit_tiny/clsANDpos/p_cls_token_expand_B_neg1_neg1.mat');

qact_embed_Width = 8;
qact_embed_Frac  = 3.0;

qact_embed_Type{1}  = numerictype('Signed',1, 'WordLength', qact_embed_Width, 'FractionLength', qact_embed_Frac); 

p_cls_token = p_cls_token_expand_B_neg1_neg1.data;
p_cls_token_1dim = reshape(p_cls_token,1,[]);

% Cat cls_token and tokens
% cat_cls_tokens_Comma_qudq_x = cat(1, double(p_cls_token_1dim), double(conv0_out_reshape_1)); 
% qudq_cat_cls_tokens_Comma_qudq_x=fi(cat_cls_tokens_Comma_qudq_x,qact_embed_Type{1}, MathType);
qudq_p_cls_token_1dim=fi(p_cls_token_1dim,qact_embed_Type{1}, MathType); %cat()can't use fixpoint number but vertcat() can use fixpoint number
qudq_cat_cls_tokens_Comma_qudq_x = vertcat(qudq_p_cls_token_1dim, conv0_out_reshape_1);

%phase_two
% LANE_NUM = 16;
LANE_NUM = 8;
qudq_p_cls_token_1dim_Integer = storedInteger(qudq_p_cls_token_1dim);
conv0_out_reshape_1_Integer = storedInteger(conv0_out_reshape_1);
conv0_out_reshape_1_reorderOutput = reorderOutput(conv0_out_reshape_1_Integer,14,14,192,8,8);
qudq_cat_cls_tokens_Comma_qudq_x_Integer = vertcat(qudq_p_cls_token_1dim_Integer, conv0_out_reshape_1_reorderOutput);

% Zero-padding operation
qudq_cat_cls_tokens_Comma_qudq_x_reshape = reshape(qudq_cat_cls_tokens_Comma_qudq_x_Integer,1,197,192);
zeros_padded = zeros(1, 13, 192);
zeros_padded=fi(zeros_padded,qact_embed_Type{1}, MathType);
zeros_padded_Integer = storedInteger(zeros_padded);
qudq_cat_cls_tokens_Comma_qudq_x_zero_padded = horzcat(qudq_cat_cls_tokens_Comma_qudq_x_reshape, zeros_padded_Integer);
qudq_cat_cls_tokens_Comma_qudq_x_reorderinputs = reorderInputs(qudq_cat_cls_tokens_Comma_qudq_x_zero_padded,15,14,192,LANE_NUM,ALIGN_SIZE);

% Decompose matrix A
qudq_cat_cls_tokens_Comma_qudq_x_reorderinputs_reshape = reshape(qudq_cat_cls_tokens_Comma_qudq_x_reorderinputs,210*LANE_NUM,192/LANE_NUM);
A = zeros(size(qudq_cat_cls_tokens_Comma_qudq_x_reorderinputs_reshape));
A=fi(A,qact_embed_Type{1}, MathType);
A_Integer = storedInteger(A);
for i = 1:LANE_NUM
        A_Integer(i, :) = qudq_cat_cls_tokens_Comma_qudq_x_reorderinputs_reshape(i, :);
end
%end of phase_two

%% Position_embedding

% position_embedding
p_pos_embed_cpu = load('./mat_extract/deit_tiny/clsANDpos/p_pos_embed_cpu.mat'); 

qact_pos_Width = 8;
qact_pos_Frac  = 3.0;%4.0

qact_pos_Type{1}  = numerictype('Signed',1, 'WordLength', qact_pos_Width, 'FractionLength', qact_pos_Frac); 

p_pos_embed_cpu_reshape = reshape(p_pos_embed_cpu.data,size(p_pos_embed_cpu.data,2),size(p_pos_embed_cpu.data,3));

qudq_p_pos_embed_cpu=fi(p_pos_embed_cpu_reshape,qact_pos_Type{1}, MathType); 

%phase_two
qudq_p_pos_embed_cpu_Integer = storedInteger(qudq_p_pos_embed_cpu);
% Zero-padding operation
qudq_p_pos_embed_cpu_reshape = reshape(qudq_p_pos_embed_cpu_Integer,1,197,192);
zeros_padded = zeros(1, 13, 192);
zeros_padded=fi(zeros_padded,qact_pos_Type{1}, MathType);
zeros_padded_Integer = storedInteger(zeros_padded);
qudq_p_pos_embed_cpu_zero_padded = horzcat(qudq_p_pos_embed_cpu_reshape, zeros_padded_Integer);
qudq_p_pos_embed_cpu_reorderinputs = reorderInputs(qudq_p_pos_embed_cpu_zero_padded,15,14,192,LANE_NUM,ALIGN_SIZE);
qudq_p_pos_embed_cpu_reorderinputs_reshape = reshape(qudq_p_pos_embed_cpu_reorderinputs,210*LANE_NUM,192/LANE_NUM);

B = zeros(size(qudq_p_pos_embed_cpu_reorderinputs_reshape));  % 
B=fi(B,qact_pos_Type{1}, MathType);
B_Integer = storedInteger(B);
for i = 1:210*LANE_NUM
        B_Integer(i, :) = qudq_p_pos_embed_cpu_reorderinputs_reshape(i, :);
end
C_Integer = A_Integer + B_Integer;

for i = 1:210*LANE_NUM
        if i<=LANE_NUM
            cls_plus_pos_210_192(i, :) = C_Integer(i, :);
        elseif i>LANE_NUM 
            cls_plus_pos_210_192(i, :) = A_Integer(i, :);
        end
end

for i = LANE_NUM+1:197*LANE_NUM
            pos_196_192(i-LANE_NUM, :) = C_Integer(i, :);
end

% Assuming this is an array you've defined in MATLAB
for i = 1:LANE_NUM
            eltwise_data(i, :) = C_Integer(i, :);
end
% eltwise_data = C_Integer; % Array contents

% Opening file for writing
fid = fopen('eltwise.h', 'wt');

% Writing macro definitions for preventing duplicate inclusion in header files
fprintf(fid, '#ifndef ELTWISE_H\n');
fprintf(fid, '#define ELTWISE_H\n\n');

% Writing the declaration of the array
fprintf(fid, 'DPTYPE_SCAL cls_pos[%d] = {\n', numel(eltwise_data));

% Writing the array data
for i = 1:numel(eltwise_data)
    fprintf(fid, '%5d, ', eltwise_data(i));
    if mod(i, LANE_NUM) == 0 % Inserting a newline after every 210 * LANE_NUM numbers to maintain a tidy format
        fprintf(fid, '\n');
    end
end

% Removing the final comma and space
fseek(fid, -2, 'cof');

% Ending array initialization
fprintf(fid, '\n};\n\n');

% Ending macro definition
fprintf(fid, '#endif // ELTWISE_H\n');

% Closing the file
fclose(fid);
%end of phase_two

qudq_x_And_qudq_pos = qudq_cat_cls_tokens_Comma_qudq_x + qudq_p_pos_embed_cpu;

% vit_qact1_Width = 16;%8
% vit_qact1_Frac  = 3.0;%3.0

vit_qact1_Width = 8;%8
vit_qact1_Frac  = 3.0;%3.0

vit_qact1_Type{1}  = numerictype('Signed',1, 'WordLength', vit_qact1_Width, 'FractionLength', vit_qact1_Frac); 

qudq_qudq_x_And_qudq_pos=fi(qudq_x_And_qudq_pos,vit_qact1_Type{1}, MathType);

%test
qudq_qudq_x_And_qudq_pos_Integer = storedInteger(qudq_qudq_x_And_qudq_pos);

% Zero-padding operation
qudq_qudq_x_And_qudq_pos_Integer_reshape = reshape(qudq_qudq_x_And_qudq_pos_Integer,1,197,192);
zeros_padded = zeros(1, 13, 192);
zeros_padded=fi(zeros_padded,vit_qact1_Type{1}, MathType);
zeros_padded_Integer = storedInteger(zeros_padded);
qudq_qudq_x_And_qudq_pos_Integer_reshape_zero_padded = horzcat(qudq_qudq_x_And_qudq_pos_Integer_reshape, zeros_padded_Integer);
qudq_qudq_x_And_qudq_pos_Integer_reorderinputs = reorderInputs(qudq_qudq_x_And_qudq_pos_Integer_reshape_zero_padded,15,14,192,8,8);
%end of test
%% save cls_token and position embedding
if(Save==1)
    % cls_token
    fid = fopen(strcat(MatOutPath,'tokens.data'), 'w');
    % fwrite(fid, storedInteger(qudq_p_cls_token_1dim), 'int8');
    
   
    % position embedding
    % fwrite(fid, storedInteger(qudq_p_pos_embed_cpu), 'int8');
    
    % reorder cls and pos
    % fwrite(fid, cls_plus_pos_210_192, 'int16');
    % fwrite(fid, pos_196_192, 'int16');

    fwrite(fid, cls_plus_pos_210_192, 'int8');  % 210 * 192 % (1680 * 4) / 192 = 210, 
    fwrite(fid, pos_196_192, 'int8');

    fclose(fid);
end
%% BLOCKS  %% BLOCKS  %% BLOCKS  %% BLOCKS  %% BLOCKS  %% BLOCKS  %% BLOCKS

qudq_blocks_0_output  = vit_blocks_0(qudq_qudq_x_And_qudq_pos);
qudq_blocks_1_output  = vit_blocks_1(qudq_blocks_0_output);
qudq_blocks_2_output  = vit_blocks_2(qudq_blocks_1_output);
qudq_blocks_3_output  = vit_blocks_3(qudq_blocks_2_output);
qudq_blocks_4_output  = vit_blocks_4(qudq_blocks_3_output);
qudq_blocks_5_output  = vit_blocks_5(qudq_blocks_4_output);
qudq_blocks_6_output  = vit_blocks_6(qudq_blocks_5_output);
qudq_blocks_7_output  = vit_blocks_7(qudq_blocks_6_output);
qudq_blocks_8_output  = vit_blocks_8(qudq_blocks_7_output);
qudq_blocks_9_output  = vit_blocks_9(qudq_blocks_8_output);
qudq_blocks_10_output = vit_blocks_10(qudq_blocks_9_output);
qudq_blocks_11_output = vit_blocks_11(qudq_blocks_10_output);

%% Norm

% norm_weight = load('./mat_extract/deit_tiny/Norm/norm_weight_fl_6.mat'); 
% norm_bias = load('./mat_extract/deit_tiny/Norm/norm_bias_fl_6.mat'); 

norm_weight = load('./mat_extract/deit_tiny/Norm/cpu_p_qudq_24_LN_weight_fl_5.mat'); 
norm_bias = load('./mat_extract/deit_tiny/Norm/cpu_p_qudq_24_LN_bias_fl_6.mat'); 

norm_weight = norm_weight.data;
norm_bias = norm_bias.data;

qudq_blocks_11_output_permute = permute(qudq_blocks_11_output, [2, 1]);
qudq_blocks_11_output_permute = single(qudq_blocks_11_output_permute);

epsilon_block_0_norm2 = 1e-6;
dim_Norm = 1;  % Because the shape is (192,197), LayerNorm is performed on the first dimension
scale_Norm = norm_weight; 
shift_Norm = norm_bias;

%----------------------------------------------------------------------
    %% phase_two
    if(Save==1)
        fid = fopen(strcat(MatOutPath,'tokens.data'), 'a'); %% 	
        % head_norm
        fwrite(fid, norm_weight, 'single');
        fwrite(fid, norm_bias, 'single');
        fclose(fid);
    end
% end of phase_two
%----------------------------------------------------------------------

% norm_qact2_output = my_layernorm_repmat(qudq_blocks_11_output_permute, epsilon_block_0_norm2, dim_Norm, scale_Norm, shift_Norm);

norm_qact2_output = int_layernorm_repmat(qudq_blocks_11_output_permute, epsilon_block_0_norm2, dim_Norm, scale_Norm, shift_Norm);

norm_qact2_output = permute(norm_qact2_output, [2, 1]);

norm_qact2_output_cls = squeeze(norm_qact2_output(1, :));

norm_qact2_Width = 8;
norm_qact2_Frac  = 4.0;

norm_qact2_Type{1}  = numerictype('Signed',1, 'WordLength', norm_qact2_Width, 'FractionLength', norm_qact2_Frac); 

qudq_norm_qact2_output_cls=fi(norm_qact2_output_cls,norm_qact2_Type{1}, MathType);

%% Head

head_weight = load('./mat_extract/deit_tiny/Head/head_weight_fl_9.mat'); 
head_bias = load('./mat_extract/deit_tiny/Head/head_bias_fl_8.mat'); 
head_weight = head_weight.data;
head_bias = head_bias.data;

head_weight_Width = 8;
head_weight_Frac  = 9.0;
head_bias_Width = 8;
head_bias_Frac  = 8.0;

head_act_out_Width = 8;
head_act_out_Frac  = 3.0;

head_weight_Type{1} = numerictype('Signed',1, 'WordLength', head_weight_Width, 'FractionLength', head_weight_Frac); 
head_bias_Type{1} = numerictype('Signed',1, 'WordLength', head_bias_Width, 'FractionLength', head_bias_Frac); 
head_act_out_Type{1} = numerictype('Signed',1, 'WordLength', head_act_out_Width, 'FractionLength', head_act_out_Frac); 

qudq_head_weight = fi(head_weight,head_weight_Type{1}, MathType); 
qudq_head_bias = fi(head_bias,head_bias_Type{1}, MathType); 

% Linear
% head = Linear(qudq_norm_qact2_output_cls, qudq_head_weight, qudq_head_bias);
 
% qudq_head_act_out_output = fi(head,head_act_out_Type{1}, MathType); 

%----------------------------------------------------------------------
%% phase_two
head = fix_fc_deit(qudq_norm_qact2_output_cls, qudq_head_weight, qudq_head_bias, norm_qact2_Type{1}, head_act_out_Type{1}, MathType);

qudq_norm_qact2_output_cls_reshape = reshape(qudq_norm_qact2_output_cls,1,1,192);
qudq_head_weight_reshape = reshape(qudq_head_weight',1,1,192,1000);
qudq_head_bias_permute = permute(qudq_head_bias,[2,1]);
head_conv = fix_conv(qudq_norm_qact2_output_cls_reshape,qudq_head_weight_reshape,qudq_head_bias_permute,1,1,0,1,norm_qact2_Type{1}, head_act_out_Type{1}, MathType);
qudq_head_act_out_output = reshape(head_conv,[],1000);

if(Save==1)
    if(OFFLINE_REORDER==0)%reorder weights on the ARM
        VEC_SIZE = 1; LANE_NUM = 1;
        fid = fopen(strcat(MatOutPath,'weights.data'), 'a');
    else
        fid = fopen(strcat(MatOutPath,sprintf('weights_vec%d_lane%d.data',VEC_SIZE,LANE_NUM)), 'a');
    end

    % % save weights and bias
    % head
    fwrite(fid, storedInteger(vectorizeWeight(qudq_head_weight_reshape, VEC_SIZE, LANE_NUM)), 'int8');
    fwrite(fid, storedInteger(vectorizeBias(qudq_head_bias_permute, LANE_NUM)), 'int8');
    fclose(fid);

    % the result of head
    disp 'Processing head  ...'
    fid = fopen(strcat(MatOutPath, 'head.data'), 'w');
    fwrite(fid, storedInteger(head_conv), 'int8');
    aa = storedInteger(head_conv);
    fclose(fid);
end
if(IsCheck==1)
    fix_check_2dim(head,qudq_head_act_out_output);
end
% end of phase_two
%----------------------------------------------------------------------

%%
searchTop = single(qudq_head_act_out_output);
% [numBetterProb, accuracy] = getAccuracy_pytorch(searchTop, label, accuracy, Batch);
[numBetterProb, accuracy] = getAccuracy_pytorch(searchTop, label, accuracy, Batch, LogFid, Iter, BatchSize, FirstImgIdx);%phase_two

%% write to log file
fprintf(LogFid, 'Pic = %d numBetterProb = %d \n\n', Iter*BatchSize+Batch+FirstImgIdx, numBetterProb);


%% 
if(Save==1)
    % % save image data
    fid = fopen(strcat(MatOutPath,'image.data'), 'w');
    fwrite(fid, storedInteger(qudq_Input_3d_permute), 'int8');
    fclose(fid);
    
    % % save results of every layer
    % patch embedding layer  
    disp 'Processing patch embedding layer   ...'
    fid = fopen(strcat(MatOutPath,'conv0_out.data'), 'w');
    fwrite(fid, storedInteger(conv0_out), 'int8');
    fclose(fid);
    
    % Concatenating CLS Token and Positional Embeddings
    disp 'Processing Concatenating CLS Token and Positional Embeddings layer   ...'
    fid = fopen(strcat(MatOutPath,'concat_x_puls_pos.data'), 'w');
    fwrite(fid, storedInteger(qudq_qudq_x_And_qudq_pos), 'int8');
    fclose(fid);
end

%%

end % end of batch 

% Temperal accuracyp
tmp_accuracy1=accuracy(1)/(BatchSize*(Iter+1)+FirstImgIdx);
tmp_accuracy5=accuracy(2)/(BatchSize*(Iter+1)+FirstImgIdx);

fprintf('Current Top-1 accuracy = %5.3f\n', tmp_accuracy1);
fprintf('Current Top-5 accuracy = %5.3f\n\n', tmp_accuracy5);
% write to log file
fprintf(LogFid, 'Current Top-1 accuracy = %5.3f\n', tmp_accuracy1);
fprintf(LogFid, 'Current Top-5 accuracy = %5.3f\n\n', tmp_accuracy5);

end % end of iteration 
fclose(LogFid); % close all opened files 
toc;

% Recording code execution time
elapsed_time = toc; % Get execution time (in seconds)
fprintf(log_fid, '%s: Program finished, elapsed time: %.6f seconds\n', ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS'), elapsed_time);

% Closing the log file
fclose(log_fid);
