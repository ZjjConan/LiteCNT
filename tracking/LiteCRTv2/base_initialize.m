function state = base_initialize(state, img)

    % do initialization
    sigma = ceil(state.scaledTargetSize ./ state.gparams.subStride) .* state.oparams.outputSigmaFactor ;
    label = generate_gaussian_label(state.gparams.featrSize, sigma, state.scaledTargetSize);
    
    img = sub_average_img(img, state.bparams.averageImage);

%     [patch, label] = data_augmenter(img, label, state);

    patch = crop_roi(img, state.targetRect, state.gparams);
    patch = aug_img(patch); patch = cat(4, patch{:});

    featr = base_extract_feature(state.net_b, patch, state.bparams);
    
%     if state.gparams.useDataAugmentation
%         for i = 1:numel(state.aparams)
%             if strcmp(state.aparams(i).type, 'dropout')
%                 featr(:,:,:,i) = dropout_featr_chns(featr(:,:,:,i));
%             end
%         end
%     end
    
    if state.hparams.useProjection & state.hparams.initUsePCA
        [r, c, d, n] = size(featr);
%         featr_ = permute(featr, [4 1 2 3]);
        featr_ = reshape(featr(:,:,:,1), r*c, d);
        pcaProjections = pca(gather(featr_));
        pcaProjections = pcaProjections(:, 1:state.hparams.projectOuDims);

        state.net_h.params(1).value(1, 1, :, :) = pcaProjections;
    end

    [state.net_h, state.trainState] = ...
        lcrt_finetune(state.net_h, featr, label, [], state.oparams, ...
                      'maxIters', state.oparams.initMaxIters, ...
                      'minIters', state.oparams.initMinIters, ...
                      'learningRate', state.oparams.initLr, ...
                      'startFrame', true, ...
                      'verbose', state.oparams.verbose);
    
   
    state.trainFeatrs{1} = featr(:,:,:,1); 
    state.trainLabels{1} = label(:,:,:,1);
    state.trainScores(1) = 1;
end

