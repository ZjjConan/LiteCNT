function state = lcrtv2_track(state, img)

    state.currFrame = state.currFrame + 1;
    
    img = sub_average_img(img, state.bparams.averageImage);
    
    [p, s] = xywh_to_ccwh(state.targetRect);
    s = bsxfun(@times, s, state.tparams.scaleFactor');
    bbox = ccwh_to_xywh(p, s);
    
    [patch, cropRatio, cropCoord] = crop_roi(img, bbox, state.gparams);
    
    featr = lcrt_extract_feature(state.net_b, patch, state.bparams);
    
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
    
    [pos, sz] = xywh_to_ccwh(state.targetRect);
    
    pos = pos + [cdelta rdelta] .* state.gparams.subStride ./ (cropRatio(sdelta, :));
%     newTargetSize = (1 - state.tparams.scaleLr) * state.targetRect(3:4) + ...
%                      state.tparams.scaleLr * state.targetRect(3:4) .* state.tparams.scaleFactor(sdelta);
 
    if state.sparams.useTSE
        state.targetRect = ccwh_to_xywh(pos, sz);
        stateScaleChange = tse_track(img, state);
        sz = (1 - state.tparams.scaleLr) * sz + state.tparams.scaleLr * sz .* state.tparams.scaleFactor(sdelta) .* stateScaleChange;  
    else
        sz = (1 - state.tparams.scaleLr) * sz + state.tparams.scaleLr * sz .* state.tparams.scaleFactor(sdelta);
    end
    
    sz = min(max(sz, state.tparams.minSize), state.tparams.maxSize);

    state.targetRect = ccwh_to_xywh(pos, sz);
    state.result(state.currFrame, :) = state.targetRect;
    
    % ---------------------
    % Prepare training data
    % ---------------------
    if (unnormed_targetScore > 0.3 || state.currFrame < 3)
        % training sample moving average maybe useful
        state.trainScores(end+1) = unnormed_targetScore;
        state.trainFeatrs{end+1} = featr(:,:,:,sdelta);
        state.trainLabels{end+1} = circshift(state.trainLabels{1}, [rdelta cdelta]);
        if numel(state.trainLabels) > state.oparams.numSamples
            if state.oparams.FIFO
                index = 2;
            else
                [~, index] = min(state.trainScores);
            end
            state.trainScores(index) = [];
            state.trainFeatrs(index) = [];
            state.trainLabels(index) = [];
        end
    end
end