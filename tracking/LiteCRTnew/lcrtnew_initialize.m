function state = lcrtnew_initialize(state, img)

%     if state.gparams.resizedRatio > 1
%         img = mexResize(img, double(state.gparams.imageSize));
%     end

%     if state.gparams.useGpu
%         img = gpuArray(single(img));
%     else
%         img = single(img);
%     end
    
    img = sub_average_img(img, state.bparams.averageImage);
    
    sigma = ceil(state.scaledTargetSize ./ state.gparams.subStride) .* state.oparams.outputSigmaFactor ;
    label = generate_gaussian_label(state.gparams.featrSize, sigma, state.scaledTargetSize);
    
%     [patch, label] = data_augmenter(img, label, state);
    
    patch = crop_roi(img, state.targetRect, state.gparams);
    patch = aug_img(patch); patch = cat(4, patch{:});
    featr = lcrt_extract_feature(state.net_b, patch, state.bparams);
    
    if state.hparams.useProjection & state.hparams.initUsePCA
        [r, c, d, n] = size(featr);
        featr_ = reshape(featr(:,:,:,1), r*c, d);
        pcaProjections = pca(gather(featr_));
        pcaProjections = pcaProjections(:, 1:state.hparams.projectOuDims);
        if state.gparams.useGpu
            state.net_h.params(1).value(1, 1, :, :) = gpuArray(pcaProjections);
        else
            state.net_h.params(1).value(1, 1, :, :) = pcaProjections;
        end
    end
    
	
    
%     a = reshape(featr_ * pcaProjections, r, c, 48);
%     state.net_h.params(3).value = a(38:58, 14:22, :) / 1e4;
    
    [state.net_h, state.trainState] = ...
        lcrt_finetune(state.net_h, featr, label, [], state.oparams, ...
                      'maxIters', state.oparams.initMaxIters, ...
                      'minIters', state.oparams.initMinIters, ...
                      'learningRate', state.oparams.initLr, ...
                      'startFrame', true, ...
                      'verbose', state.oparams.verbose);
   
    state.trainFeatrs{1} = featr(:,:,:,1); 
    state.trainLabels{1} = label;
    state.trainScores(1) = 1;
end

