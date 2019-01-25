function state = lcrt_track(state, img)

    state.currFrame = state.currFrame + 1;
    
    % bug in cvpr submission
    if state.gparams.resizedRatio ~= 1
        % image size formated as [x y]
        img = mexResize(img, state.gparams.imageSize([2,1]), 'linear');
    end
    
    if state.gparams.useGpu
        img = gpuArray(single(img));
    else
        img = single(img);
    end

    img = sub_average_img(img, state.bparams.averageImage);
    
    [p, s] = xywh_to_ccwh(state.targetRect);
    s = bsxfun(@times, s, state.tparams.scaleFactor');
    bbox = ccwh_to_xywh(p, s);
    
    [patch, cropRatio] = crop_roi(img, bbox, state.gparams);
    featr = lcrt_extract_feature(state.net_b, patch, state.bparams);
    
    % ----------
    % Estimation
    % ----------
    if state.gparams.useGpu
        featr = gpuArray(featr); 
    end
    state.net_h.eval({'input', featr});
    predictions = gather(state.net_h.vars(state.hparams.netOutIdx).value);
    predictions = predictions .* state.tparams.motionWindow;
%     show_response(patch, predictions, 0.3, state);
    % multi-scale scores
    targetScore = max(reshape(predictions, [], size(patch, 4)));
    [targetScore, sdelta] = max(targetScore);
    [rdelta, cdelta] = find(predictions(:,:,:,sdelta) == targetScore, 1);
  
    rdelta = mean(rdelta) - ceil(state.gparams.featrSize(2)/2);
    cdelta = mean(cdelta) - ceil(state.gparams.featrSize(1)/2);
    state.targetScore = targetScore;
    state.targetRect(1:2) = state.targetRect(1:2) + [cdelta rdelta] .* state.gparams.subStride ./ cropRatio(sdelta, :);
    newTargetSize = (1 - state.tparams.scaleLr) * state.targetRect(3:4) + ...
                     state.tparams.scaleLr * state.targetRect(3:4) .* state.tparams.scaleFactor(sdelta);
    state.targetRect(3:4) = min(max(newTargetSize, state.tparams.minSize), state.tparams.maxSize);
    
    state.result(state.currFrame, :) = state.targetRect .* state.gparams.resizedRatio;
    
    % ---------------------
    % Prepare training data
    % ---------------------
    if targetScore > 0.3 || state.currFrame < 3
        % training sample moving average maybe useful
        state.trainScores(end+1) = targetScore;
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