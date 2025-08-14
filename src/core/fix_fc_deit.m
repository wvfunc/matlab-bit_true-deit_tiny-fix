%Fully Connected Layer.
%bottom is a 2d matrix: N x 1.
%top is a 2d matrix: M x 1.
%weight is a 4d matrix: 1 x 1 x N x M.
%bias is a 4d matrix: 1 x 1 x 1 x M.
%Formula: top=weights'*bottom+bias.
function [ top ] = fix_fc_deit( bottom, weight, bias, DataInType, DataOutType, MathType )
    %[~,~,N,M]=size(weight);
    %weightFlattened=reshape(weight, [N, M]);
    %biasFlattened=reshape(bias, [M, 1]);
    %top=weightFlattened'*bottom+biasFlattened;
    if(bottom.FractionLength~=DataInType.FractionLength)
		error('Output data format does not match input data !!!');
    end
	%bottom=fi(bottom, DataInType, MathType);
    %top=weight'*bottom+bias;
	%top=fi(top, DataOutType, MathType);
    % Changed to be consistent with fix_conv
    weight = single(weight);
    bias = single(bias);
    top = single(bottom)*weight' + bias;
    
    % Scheme-1 Two step rounding
    %top=fi(top, DataOutType, MathType); % first perform rounding, then add the bias
    %top=fi(top+bias, DataOutType, MathType);
    % Scheme-2 One step rounding
    %top=fi(top+bias, DataOutType, MathType);
    
    top = fi(top, DataOutType, MathType); % first perform rounding, then add the bias

end
