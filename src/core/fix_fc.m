%Fully Connected Layer.
%bottom is a 2d matrix: N x 1.
%top is a 2d matrix: M x 1.
%weight is a 4d matrix: 1 x 1 x N x M.
%bias is a 4d matrix: 1 x 1 x 1 x M.
%Formula: top=weights'*bottom+bias.
function [ top ] = fix_fc( bottom, weight, bias, DataInType, DataOutType, MathType )
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
    top=weight'*bottom;
    
    % Scheme-1 Two Step Rounding -> use both step-1 and 2
    % Scheme-2 One Step Rounding -> use only step-2
    
    % Step-1 rounding
    %top=fi(top, DataOutType, MathType); % first perform rounding, then add the bias
    
    % Add bias.
    top = top+bias;
    
    % Step-2 rounding
    top=fi(top, DataOutType, MathType);
end
