function state = lcrtv2_state_initialize(img, region, opts)
    % init network for feature extraction    
    net_b = init_featrnet(opts.bparams);
    
    % contraint the target size for efficient tracking
    if max(region(3:4)) > opts.gparams.maxTargetSize
        resizedRatio = max(region(3:4)) ./ opts.gparams.maxTargetSize;
    elseif max(region(3:4)) < opts.gparams.minTargetSize
        resizedRatio = max(region(3:4)) ./ opts.gparams.minTargetSize;
    else
        resizedRatio = 1;
    end

    orgTargetSize = region(3:4);
    scaledTargetSize = round(region(3:4) ./ resizedRatio);
    
    % determine the output size and subsampling factor
    switch opts.gparams.inputShape
        case 'square'
            inputSize = round(repmat(sqrt(prod(scaledTargetSize * (1 + opts.gparams.searchPadding))), 1, 2));
        case 'proportional'
            inputSize = round(scaledTargetSize * (1 + opts.gparams.searchPadding));
    end

    opts.bparams.cosineWindow = [];
    varSizes = net_b.getVarSizes({'input',[inputSize 3 1]});
    lastLayerSize = varSizes{end}(1:2);
    featrSize = lastLayerSize + 1 + mod(lastLayerSize, 2);

    while featrSize(1) > lastLayerSize(1)
        inputSize = inputSize + [1, 0];
        varSizes = net_b.getVarSizes({'input',[inputSize 3 1]});
        lastLayerSize = varSizes{end}(1:2);
    end
    while featrSize(2) > lastLayerSize(2)
        inputSize = inputSize + [0, 1];
        varSizes = net_b.getVarSizes({'input',[inputSize 3 1]});
        lastLayerSize = varSizes{end}(1:2);
    end
    opts.bparams.cosineWindow = single(hann(lastLayerSize(2)) * hann(lastLayerSize(1))');
    
    inputSize = get_input_size(net_b, featrSize);
   
    subStride = inputSize ./ featrSize;
    
    if strcmpi(opts.gparams.inputShape, 'square')
        opts.gparams.searchPadding = inputSize ./ scaledTargetSize - 1;
    end

    % init head --------------------------------------------
    switch lower(opts.hparams.headType)
        case 'maskconv'
            net_h = init_det_mconv(scaledTargetSize ./ subStride, opts.hparams);
        case 'baseconv'
            net_h = init_det_nconv(scaledTargetSize ./ subStride, opts.hparams);
        case 'twobranch'
            net_h = init_det_2conv(scaledTargetSize ./ subStride, opts.hparams);
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
    
    motionWindow =  single(hann(featrSize(1)) * hann(featrSize(2))');
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
        maxScaleFactor = opts.tparams.scaleStep ^ floor(log(min(imageSize ./ orgTargetSize)) / log(opts.tparams.scaleStep));
    end
    
    if opts.gparams.useDataAugmentation
        aparams = struct();
        ct = 1;
        for i = 1:length(opts.aparams)
            if length(opts.aparams(i).param) > 1
                for j = 1:length(opts.aparams(i).param)
                    aparams(ct).type = opts.aparams(i).type;
                    aparams(ct).param = opts.aparams(i).param{j};
                    ct = ct + 1;
                end
            else
                aparams(ct).type = opts.aparams(i).type;
                aparams(ct).param = opts.aparams(i).param;
                ct = ct + 1;
            end
        end
    else
        aparams = [];
    end
    
    
    opts.gparams.gridGenerator = gridGenerator;
    opts.gparams.resizedRatio = resizedRatio;
    opts.gparams.imageSize = imageSize;
    opts.gparams.inputSize = inputSize;
    opts.gparams.subStride = subStride;
    opts.gparams.featrSize = featrSize;
    
    opts.tparams.motionWindow = motionWindow;
    opts.tparams.scaleFactor = scaleFactor;
    opts.tparams.minSize = max(5, minScaleFactor .* orgTargetSize);
    opts.tparams.maxSize = min(imageSize, maxScaleFactor .* orgTargetSize);
    
    opts.hparams.netOutIdx = state.net_h.getVarIndex('prediction');
    
    state.aparams = aparams;
    state.gparams = opts.aparams;
    state.gparams = opts.gparams;
    state.bparams = opts.bparams;
    state.hparams = opts.hparams;
    state.tparams = opts.tparams;
    state.oparams = opts.oparams;  
    
    state.result = region;    
    state.targetRect = region;
    state.scaledTargetSize = scaledTargetSize;
    
    state.targetScore = 1; 
    state.currFrame = 1;
    state.succIndex = 1;
end