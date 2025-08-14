function output = ViT_gelu(x)
    x = single(x);
    output = 0.5 * x .* (1 + erf(x ./ sqrt(2)));
end
