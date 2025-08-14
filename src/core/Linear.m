function output = Linear(input, weight, bias)
    input = single(input);
    weight = single(weight);
    bias = single(bias);
    % input parameter:
    % input: input tensor of size [batch size, input size]
    % weight: Weight matrix of size [output size, input size]
    % bias: Bias vector of size [output size]

    % Computing linear functions
    output = input * weight' + bias; 
end
