function weights = init_weights(sz, bias, scale)
    if nargin < 3
        scale = 0.01;
    end
    if nargin < 2 || bias
        weights{1} = scale * randn(sz, 'single');
        weights{2} = zeros(sz(4), 1, 'single');
    else
        weights{1} = scale * randn(sz, 'single');
    end
end