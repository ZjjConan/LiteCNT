function state = lcnt_initialize(state, img)

    img = sub_average_img(img, state.bparams.averageImage);
    
    sigma = ceil(state.scaledTargetSize ./ state.gparams.subStride) .* state.oparams.outputSigmaFactor ;
    label = generate_gaussian_label(state.gparams.featrSize, sigma, state.scaledTargetSize);
       
    patch = crop_roi(img, state.targetRect, state.gparams);
    patch = aug_img(patch); patch = cat(4, patch{:});
    featr = lcnt_extract_feature(state.net_b, patch, state.bparams);
    
    if state.hparams.useProjection & state.hparams.initUsePCA
        [r, c, d, n] = size(featr);
        featr_ = reshape(featr(:,:,:,1), r*c, d);
        pcaProjections = pca(gather(featr_));
        pcaProjections = pcaProjections(:, 1:state.hparams.projectOuDims);

        if state.gparams.useGpu
            state.net_h.params(1).value(1,1,:,:) = gpuArray(pcaProjections);
        else
            state.net_h.params(1).value(1,1,:,:) = pcaProjections;
        end
    end

    
    state.net_h = ...
        lcnt_finetune(state.net_h, featr, label, [], state.oparams, ...
                      'maxIters', state.oparams.initMaxIters, ...
                      'minIters', state.oparams.initMinIters, ...
                      'learningRate', state.oparams.initLr, ...
                      'startFrame', true, ...
                      'verbose', state.oparams.verbose);
   
    state.trainFeatrs{1} = featr(:,:,:,1); 
    state.trainLabels{1} = label(:,:,:,1);
    state.trainScores(1) = 1;
end

