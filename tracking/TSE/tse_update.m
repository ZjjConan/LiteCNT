function state = tse_update(img, state)
    % Get scale filter features
%     scales = state.tseFilter.scaleSizeFactors;
    
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
%     featr = reshape(featr, [], size(featr, 4));
   
    if state.currFrame == 1
        state.tsEstimator.featr = featr;
    else
        state.tsEstimator.featr = (1 - state.sparams.TSELearningRate) * state.tsEstimator.featr + state.sparams.TSELearningRate * featr;
    end
    
    for i = 1:numEstimator
        fstart = (i-1) * state.sparams.numScales + 1;
        fend = i * state.sparams.numScales;
        
        currFeatr = state.tsEstimator.featr(:,fstart:fend);
        
        % Compute projection basis
        bigY = currFeatr;
        bigY_den = featr(:,fstart:fend);

        [state.tsEstimator.basis{i}, ~] = qr(bigY, 0);
        [scale_basis_den, ~] = qr(bigY_den, 0);

        state.tsEstimator.basis{i} = state.tsEstimator.basis{i}';

        % Compute numerator
        sf_proj = fft(feature_projection_scale(state.tsEstimator.featr(:,fstart:fend), state.tsEstimator.basis{i}, state.tsEstimator.window), [], 2);
        state.tsEstimator.sf_num{i} = bsxfun(@times, state.tsEstimator.yf, conj(sf_proj));

        % Update denominator
%         featr = feature_projection_scale(featr, scale_basis_den',  state.tsEstimator.window);
        xsf = fft(feature_projection_scale(featr(:,fstart:fend), scale_basis_den',  state.tsEstimator.window), [], 2);
        new_sf_den = sum(xsf .* conj(xsf),1);
        if state.currFrame == 1
            state.tsEstimator.sf_den{i} = new_sf_den;
        else
            state.tsEstimator.sf_den{i} = (1 - state.sparams.TSELearningRate) * state.tsEstimator.sf_den{i} + state.sparams.TSELearningRate * new_sf_den;
        end
    end
end

