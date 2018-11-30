function g = generate_bilinear_grids(p, s, opts)
    % bbox [x y w h]
    n = size(p, 2);
    p = p - 1;
    im_h = opts.imageSize(1) - 1;
    im_w = opts.imageSize(2) - 1;
    
    if opts.useGpu
        g = gpuArray.zeros(1, 1, 6, n, 'single');
    else
        g = zeros(1, 1, 6, n, 'single');
    end
    
    g(1, 1, 1, :) = s(2, :) ./ im_h;
    g(1, 1, 4, :) = s(1, :) ./ im_w;
    g(1, 1, 5, :) = p(2, :) * 2 / im_h - 1;
    g(1, 1, 6, :) = p(1, :) * 2 / im_w - 1;
    
    g = opts.gridGenerator.forward({g});
    g = g{1};
    g = max(-1, g);
    g = min(1, g);
end
