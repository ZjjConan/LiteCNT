function state = lcrtup_track(state, img)

    state.currFrame = state.currFrame + 1;

    if state.resizedRatio > 1
        img = mexResize(img, state.opts.imageSize);
    end
    
    if state.opts.useGpu
        img = gpuArray(single(img));
    else
        img = single(img);
    end
    
    img = sub_average_img(img, state.opts);
    
    [p, s] = xywh_to_ccwh(state.targetRect);
    s = bsxfun(@times, s, state.scaleFactor');
    bbox = ccwh_to_xywh(p, s);
    
    [patch, cropRatio] = crop_roi(img, bbox, state.opts);
    
    
    % ----------
    % Estimation
    % ----------
    if state.opts.useGpu, patch = gpuArray(patch); end
    state.net_d.mode = 'test';
    state.net_d.eval({'input', patch});
    predictions = gather(state.net_d.vars(state.netOutIdx).value);
    predictions = predictions .* state.motionWindow;
    
    % multi-scale scores
    targetScore = max(reshape(predictions, [], state.opts.numScales));
    [targetScore, sdelta] = max(targetScore);
    [rdelta, cdelta] = find(predictions(:,:,:,sdelta) == targetScore, 1, 'first');
  
    rdelta = rdelta - ceil(state.featrSize(2)/2);
    cdelta = cdelta - ceil(state.featrSize(1)/2);
    state.targetScore = targetScore;
    state.targetRect(1:2) = state.targetRect(1:2) + [cdelta rdelta] .* state.opts.subStride ./ cropRatio(sdelta, :);
    newTargetSize = (1 - state.opts.scaleLr) * state.targetRect(3:4) + ...
                     state.opts.scaleLr * state.targetRect(3:4) .* state.scaleFactor(sdelta);
    state.targetRect(3:4) = min(max(newTargetSize, state.minSize), state.maxSize);
    
    state.result(state.currFrame, :) = state.targetRect .* state.resizedRatio;
    
    % ---------------------
    % Prepare training data
    % ---------------------
    if (targetScore > 0.35 | state.currFrame < 3)
        state.trainScores(end+1) = targetScore;
        state.trainFeatrs{end+1} = patch(:,:,:,sdelta);
        state.trainLabels{end+1} = circshift(state.trainLabels{1}, [rdelta cdelta]);
        if numel(state.trainLabels) > state.opts.numSamples
            if state.opts.FIFO
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