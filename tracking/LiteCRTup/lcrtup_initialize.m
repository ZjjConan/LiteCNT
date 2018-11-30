function state = lcrtup_initialize(state, img)

    if state.resizedRatio > 1
        img = mexResize(img, double(state.opts.imageSize));
    end

    if state.opts.useGpu
        img = gpuArray(single(img));
    else
        img = single(img);
    end

    img = sub_average_img(img, state.opts);
    
    patch = crop_roi(img, state.targetRect, state.opts);
    
    if state.opts.useGpu, patch = gpuArray(patch); end
    state.net_d.mode = 'test';
    state.net_d.eval({'input', patch(:,:,:,1)});
    featr = state.net_d.vars(state.pcaOutIdx).value;
    [r, c, d, n] = size(featr);
    featr_ = reshape(featr(:,:,:,1), r*c, d);
    pcaProjections = pca(gather(featr_));
    pcaProjections = pcaProjections(:, 1:state.opts.PCADims);
    
    state.net_d.params(3).value(1, 1, :, :) = pcaProjections;
    
	sigma = ceil(state.targetRect(:, 3:4) ./ state.opts.subStride) .* state.opts.outputSigmaFactor ;
    label = generate_gaussian_label(state.featrSize, sigma, state.targetRect(:, 3:4));

    patch = aug_img(patch); patch = cat(4, patch{:});
        
    state.net_d = lcrt_finetune(state.net_d, patch, label, [], state.opts, ...
                                'maxIters', state.opts.initMaxIters, ...
                                'minIters', state.opts.initMinIters, ...
                                'learningRate', state.opts.initLr, ...
                                'startFrame', true);
    state.net_d.params(1).learningRate = 0;
    state.net_d.params(2).learningRate = 0; 
    state.trainFeatrs{1} = patch(:,:,:,1); 
    state.trainLabels{1} = label;
    state.trainScores(1) = 1;
end

