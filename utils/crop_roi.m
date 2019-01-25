function [patch, cropRatio, cropCoord] = crop_roi(image, bbox, opts)
    % boxes(1, :) is the previously estimated location or ground truth points
    % patch cropped image patch using bilinear interpolation
    % cinfo cropped information start and end coordinate
    
    
    p = bbox(:, 1:2) + bbox(:, 3:4)/2;
    s = bbox(:, 3:4) .* (1 + opts.searchPadding);
    
    cropCoord = [p-s/2 p+s/2];
    cropRatio = bsxfun(@rdivide, opts.inputSize, s);

    grids = generate_bilinear_grids(p', s', opts); 

    if opts.useGpu
        image = gpuArray(image);
        grids = gpuArray(grids);
    end

    patch = vl_nnbilinearsampler(image, grids);

end