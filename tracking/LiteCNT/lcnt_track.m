function state = lcnt_track(state, img)

    state.currFrame = state.currFrame + 1;
    
    img = sub_average_img(img, state.bparams.averageImage);
    
    [pos, sz] = xywh_to_ccwh(state.targetRect);
    msz = bsxfun(@times, sz, state.tparams.scaleFactor');
    bbox = ccwh_to_xywh(pos, msz);
    
    [patch, cropRatio] = crop_roi(img, bbox, state.gparams);
    
    featr = lcnt_extract_feature(state.net_b, patch, state.bparams);
    
    % ----------
    % Estimation
    % ----------
    if state.gparams.useGpu
        featr = gpuArray(featr); 
    end
    state.net_h.eval({'input', featr});
    predictions = gather(state.net_h.vars(state.hparams.netOutIdx).value);
    predictions = bsxfun(@times, predictions, state.tparams.scalePenalty);
    targetScore = max(reshape(predictions, [], size(patch, 4)));
    [unnormed_targetScore, sdelta] = max(targetScore);
    predictions = predictions(:, :, :, sdelta);
    predictions = predictions - min(predictions(:));
    predictions = predictions / sum(predictions(:));
    predictions = (1 - state.tparams.motionSigmaFactor) * predictions + state.tparams.motionSigmaFactor * state.tparams.motionWindow;
    targetScore = max(predictions(:));
    [rdelta, cdelta] = find(predictions == targetScore);

    rdelta = mean(rdelta) - ceil(state.gparams.featrSize(2)/2);
    cdelta = mean(cdelta) - ceil(state.gparams.featrSize(1)/2);
    state.targetScore = unnormed_targetScore;
 
    pos = pos + [cdelta rdelta] .* state.gparams.subStride ./ (cropRatio(sdelta, :));
 
    sz = (1 - state.tparams.scaleLr) * sz + state.tparams.scaleLr * sz .* state.tparams.scaleFactor(sdelta);
    
    sz = min(max(sz, state.tparams.minSize), state.tparams.maxSize);
    
    state.targetRect = ccwh_to_xywh(pos, sz);
    state.result(state.currFrame, :) = state.targetRect;
    
    % ---------------------
    % Prepare training data
    % ---------------------
    if (unnormed_targetScore > 0.35 || state.currFrame < 3)
        % training sample moving average maybe useful
        state.trainScores(end+1) = unnormed_targetScore;
        state.trainFeatrs{end+1} = featr(:,:,:,sdelta);
        state.trainLabels{end+1} = circshift(state.trainLabels{1}, [rdelta cdelta]);
        if numel(state.trainLabels) > state.oparams.numSamples
            state.trainScores(2) = [];
            state.trainFeatrs(2) = [];
            state.trainLabels(2) = [];
        end
    end
end