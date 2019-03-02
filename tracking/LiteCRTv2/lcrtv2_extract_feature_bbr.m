function featr_bbr = lcrtv2_extract_feature_bbr(bbox, featr, opts)

    p = bbox(:, 1:2) + bbox(:, 3:4)/2;
    s = bbox(:, 3:4);
    
    grids = generate_bilinear_grids(p', s', opts); 

    if opts.useGpu
        featr = gpuArray(featr);
        grids = gpuArray(grids);
    end
    featr_bbr = vl_nnbilinearsampler(featr, grids);
end

