function state = lcnt_state_initialize(img, region, opts)
    % init network for feature extraction    
    net_b = init_featrnet(opts.bparams);
    
    % contraint the target size for efficient tracking
    if max(region(3:4)) > opts.gparams.maxTargetSize
        scaledRatio = max(region(3:4)) ./ opts.gparams.maxTargetSize;
    elseif max(region(3:4)) < opts.gparams.minTargetSize
        scaledRatio = max(region(3:4)) ./ opts.gparams.minTargetSize;
    else
        scaledRatio = 1;
    end

    orgTargetSize = region(3:4);
    scaledTargetSize = round(region(3:4) ./ scaledRatio);
    
    inputSize = round(repmat(sqrt(prod(scaledTargetSize * (1 + opts.gparams.searchPadding))), 1, 2));
    
    opts.bparams.cosineWindow = [];
    varSizes = net_b.getVarSizes({'input',[inputSize 3 1]});
    featrSize = varSizes{end}(1:2);
    if mod(featrSize(1), 2) == 0, featrSize(1) = featrSize(1) + 1; end
    if mod(featrSize(2), 2) == 0, featrSize(2) = featrSize(2) + 1; end
    
    opts.bparams.cosineWindow = single(hann(featrSize(2)) * hann(featrSize(1))');
    
    inputSize = get_input_size(net_b, featrSize);
    
    subStride = inputSize ./ featrSize;
    
    opts.gparams.searchPadding = inputSize ./ scaledTargetSize - 1;

    % init head --------------------------------------------
    switch lower(opts.hparams.headType)
        case 'amrconv'
            net_h = init_det_amrconv(scaledTargetSize ./ subStride, opts.hparams);
        case 'baseconv'
            net_h = init_det_baseconv(scaledTargetSize ./ subStride, opts.hparams);
        case 'nmrconv'
            net_h = init_det_nmrconv(scaledTargetSize ./ subStride, opts.hparams);
    end
    % ---------------------------------------------------------
     
    state.paramSize = get_model_size(net_b);
    state.paramSize = state.paramSize + get_model_size(net_h);
    state.net_b = net_b;
    state.net_h = net_h;
    if opts.gparams.useGpu
        state.net_b.move('gpu');
        state.net_h.move('gpu');
        opts.bparams.averageImage = gpuArray(opts.bparams.averageImage);
    end
    
    motionWindow =  single(hann(featrSize(2)) * hann(featrSize(1))');
    motionWindow = motionWindow / sum(motionWindow(:));    
    
    imageSize = [size(img, 2), size(img, 1)];
    gridGenerator = dagnn.AffineGridGenerator('Ho', inputSize(2), 'Wo', inputSize(1));
   
    numScales = opts.tparams.numScales;
    scaleFactor = (-floor(numScales-1)/2):ceil((numScales-1)/2);
    scaleFactor = opts.tparams.scaleStep .^ scaleFactor;
    scalePenalty = repmat(opts.tparams.scalePenalty, numScales, 1);
    scalePenalty(ceil(numScales/2)) = 1;
    opts.tparams.scalePenalty = reshape(scalePenalty, 1, 1, 1, numScales);

    if numScales > 0
        minScaleFactor = opts.tparams.scaleStep ^ ceil(log(max(5 ./ inputSize)) / log(opts.tparams.scaleStep));
        maxScaleFactor = opts.tparams.scaleStep ^ floor(log(min(imageSize ./ scaledTargetSize)) / log(opts.tparams.scaleStep));
    end
    
    opts.gparams.gridGenerator = gridGenerator;
    opts.gparams.resizedRatio = scaledRatio;
    opts.gparams.imageSize = imageSize;
    opts.gparams.inputSize = inputSize;
    opts.gparams.subStride = subStride;
    opts.gparams.featrSize = featrSize;
    
    opts.tparams.motionWindow = motionWindow;
    opts.tparams.scaleFactor = scaleFactor;
    opts.tparams.minSize = max(5, minScaleFactor .* orgTargetSize);
    opts.tparams.maxSize = min(imageSize, maxScaleFactor .* orgTargetSize);
    
    opts.hparams.netOutIdx = state.net_h.getVarIndex('prediction');
    
    state.gparams = opts.gparams;
    state.bparams = opts.bparams;
    state.hparams = opts.hparams;
    state.tparams = opts.tparams;
    state.oparams = opts.oparams;  
    
    state.result = region;    
    state.targetRect = region;
    state.scaledTargetSize = scaledTargetSize;
    state.scaledRatio = scaledRatio;
    
    state.targetScore = 1; 
    state.currFrame = 1;
end