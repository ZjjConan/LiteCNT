function state = lcrtv2_initialize(state, img)
    
    img = sub_average_img(img, state.bparams.averageImage);
    
    sigma = ceil(state.scaledTargetSize ./ state.gparams.subStride) .* state.oparams.outputSigmaFactor ;
    label = generate_gaussian_label(state.gparams.featrSize, sigma, state.scaledTargetSize);
       
%     [patch, label] = data_augmenter(img, label, state);
    [patch, cropRatio, cropCoord] = crop_roi(img, state.targetRect, state.gparams);
    patch = aug_img(patch); patch = cat(4, patch{:});
    featr = lcrt_extract_feature(state.net_b, patch, state.bparams);
    
    % dropout features
%     if state.gparams.useDataAugmentation
%         for i = 1:numel(state.aparams)
%             if strcmp(state.aparams(i).type, 'dropout')
%                 featr(:,:,:,i) = dropout_featr_chns(featr(:,:,:,i));
%             end
%         end
%     end
    
    % bbr related
    if state.tparams.useBBR
        samples = generate_samples('uniform_aspect', state.targetRect, state.tparams.BBRInitNums * 10, ...
            state.gparams.imageSize, state.tparams.BBRScaleFactor, 0.3, 10);
        r = overlap_ratio(samples, state.targetRect);
        samples = samples(r>0.6, :);
        samples = samples(randsample(end, min(state.tparams.BBRInitNums, end)), :);
        
        gridGenerator = dagnn.AffineGridGenerator('Ho', state.tparams.BBRFeatrSize(2), ...
                                                  'Wo', state.tparams.BBRFeatrSize(1));
        state.tparams.gridGenerator = gridGenerator;
        state.tparams.useGpu = state.gparams.useGpu;
        state.tparams.searchPadding = 0;
        state.tparams.imageSize = state.gparams.featrSize;
        
        projected_samples = project_bbox(samples, cropCoord, cropRatio);
        projected_samples(:, 1:2) = projected_samples(:, 1:2) ./ state.gparams.subStride;
        projected_samples(:, 3:4) = projected_samples(:, 3:4) ./ state.gparams.subStride;
        featr_bbr = crop_roi(featr(:,:,:,1), projected_samples, state.tparams);
        
        X = permute(gather(featr_bbr), [4,3,1,2]);
        X = X(:,:);
        bbox = samples;
        gts = repmat(state.targetRect, size(samples, 1), 1);
        state.BBR = train_bbox_regressor(X, bbox, gts);
        state.paramSize = state.paramSize + 4 * (numel(state.BBR.model.mu) + numel(state.BBR.model.T) + ...
            numel(state.BBR.model.T_inv) + numel(state.BBR.model.Beta));
    end

    
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

