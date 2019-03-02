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
%     if state.gparams.useGpu
%         featr = gpuArray(featr); 
%     end
%     state.net_h.eval({'input', featr});
%     predictions = gather(state.net_h.vars(state.hparams.netOutIdx).value);
%     predictions = predictions .* state.tparams.motionWindow;
% %     show_response(patch, predictions, 0.3, state);
%     % multi-scale scores
%     targetScore = max(reshape(predictions, [], size(patch, 4)));
%     [targetScore, sdelta] = max(targetScore);
%     [rdelta, cdelta] = find(predictions(:,:,:,sdelta) == targetScore, 1);

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
    state.targetRect(1:2) = state.targetRect(1:2) + [cdelta rdelta] .* state.gparams.subStride ./ (cropRatio(sdelta, :));
    newTargetSize = (1 - state.tparams.scaleLr) * state.targetRect(3:4) + ...
                     state.tparams.scaleLr * state.targetRect(3:4) .* state.tparams.scaleFactor(sdelta);
    state.targetRect(3:4) = min(max(newTargetSize, state.tparams.minSize), state.tparams.maxSize);
    
    
    if (state.tparams.useBBR && unnormed_targetScore > 0.7)
        samples = generate_samples('gaussian', state.targetRect, 100, state.gparams.imageSize, ...
            state.tparams.BBRScaleFactor, 0.05, 0.05);
        r = overlap_ratio(samples, state.targetRect);
        [r, i] = sort(r, 'descend');
        a = find(r > 0.7);
        a = a(1:min(numel(a), 5));
        if ~all(a == 0) 
            samples = samples(i(a), :);
            projected_samples = project_bbox(samples, cropCoord(sdelta, :), cropRatio(sdelta, :));  
            projected_samples(:, 1:2) = projected_samples(:, 1:2) ./ state.gparams.subStride;
            projected_samples(:, 3:4) = projected_samples(:, 3:4) ./ state.gparams.subStride;

            featr_bbr = crop_roi(featr(:,:,:,sdelta), projected_samples, state.tparams);

            X = permute(gather(featr_bbr), [4,3,1,2]);
            X = X(:,:);
            bbr_samples = predict_bbox_regressor(state.BBR.model, X, samples);
            state.targetRect = mean(bbr_samples, 1);
        end
    end
    
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