function stateChangeFactor = tse_track(img, state)

    [pos, sz] = xywh_to_ccwh(state.targetRect);
    
    sz_src = sz;
    sz_chn = bsxfun(@times, sz_src, state.tsEstimator.scaleSizeFactors');

    if state.sparams.useWEstimator
        sz_new{1} = [sz_chn(:, 1), repmat(sz_src(2), state.sparams.numScales, 1)];
    end
    
    if state.sparams.useHEstimator
        sz_new{2} = [repmat(sz_src(1), state.sparams.numScales, 1), sz_chn(:, 2)];
    end
    
    numEstimator = numel(sz_new);

    bbox = ccwh_to_xywh(pos, cat(1, sz_new{:}));
    
    patch = crop_roi(img, bbox, state.sparams);
    
    featr = tse_extract_feature(state.net_b, patch, state.sparams);
    
    stateChangeFactor = zeros(1, numEstimator);
    
    for i = 1:numEstimator
        fstart = (i - 1) * state.sparams.numScales + 1;
        fend = i * state.sparams.numScales;
        
        xs = feature_projection_scale(featr(:,fstart:fend), state.tsEstimator.basis{i}, state.tsEstimator.window);

        xsf = fft(xs, [], 2);
        response = sum(state.tsEstimator.sf_num{i} .* xsf, 1) ./ (state.tsEstimator.sf_den{i} + state.sparams.TSELambda);
        response_full = ifft(resizeDFT(response, state.sparams.TSENumInterpScales), 'symmetric');

        full_tse_index = find(response_full == max(response_full(:)), 1);

        id1 = mod(full_tse_index -1 -1,state.sparams.TSENumInterpScales)+1;
        id2 = mod(full_tse_index +1 -1,state.sparams.TSENumInterpScales)+1;

        poly_x = [state.tsEstimator.interpScaleFactors(id1), state.tsEstimator.interpScaleFactors(full_tse_index), state.tsEstimator.interpScaleFactors(id2)];
        poly_y = [response_full(id1), response_full(full_tse_index), response_full(id2)];

        poly_A_mat = [poly_x(1)^2, poly_x(1), 1;...
                poly_x(2)^2, poly_x(2), 1;...
                poly_x(3)^2, poly_x(3), 1 ];

        poly = poly_A_mat\poly_y';

        stateChangeFactor(i) = -poly(2)/(2*poly(1));
    end
end

