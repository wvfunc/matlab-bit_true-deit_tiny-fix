function output = ViTSoftmax(input, dim)
    input = single(input);
    
    % Retrieves the maximum value of the specified dimension
    maxVals = max(input, [], dim);
    
    % Subtracts the maximum value from the specified dimension
    shiftedInput = input - maxVals;
    
    % Calculating index values
    expVals = exp(shiftedInput);
    
    % Sum over the specified dimension
    sumExp = sum(expVals, dim);
    
    % Dividing by the sum yields the softmax result
    output = expVals ./ sumExp;
end
