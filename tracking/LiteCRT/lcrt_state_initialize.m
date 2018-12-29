function state = lcrt_state_initialize(img, region, opts)
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
    targetSize = round(region(3:4) ./ resizedRatio); 
    
    % determine the output size and subsampling factor
    switch opts.gparams.inputShape
        case 'square'
            inputSize = round(repmat(sqrt(prod(targetSize * (1 + opts.gparams.searchPadding))), 1, 2));
        case 'proportional'
            inputSize = round(targetSize * (1 + opts.gparams.searchPadding));
    end
    
    opts.bparams.cosineWindow = [];
    featr = lcrt_extract_feature(net_b, randn([inputSize([2,1]) 3], 'single'), opts.bparams);
    featrSize = [size(featr,2) size(featr,1)];
    if mod(featrSize(1), 2) == 0, featrSize(1) = featrSize(1) + 1; end
    if mod(featrSize(2), 2) == 0, featrSize(2) = featrSize(2) + 1; end
    opts.bparams.cosineWindow = single(hann(featrSize(2)) * hann(featrSize(1))');
    
    inputSize = get_input_size(net_b, featrSize);
%     inputSize = round(featrSize .* substride);
%     varSizes = net_b.getVarSizes({'input',[inputSize 3 1]});
%     lastLayerSize = varSizes{end}(1:2);
% %     
%     while featrSize(1) > lastLayerSize(1)
%         inputSize = inputSize + [1, 0];
%         varSizes = net_b.getVarSizes({'input',[inputSize 3 1]});
%         lastLayerSize = varSizes{end}(1:2);
%     end
%     while featrSize(2) > lastLayerSize(2)
%         inputSize = inputSize + [0, 1];
%         varSizes = net_b.getVarSizes({'input',[inputSize 3 1]});
%         lastLayerSize = varSizes{end}(1:2);
%     end
%     opts.bparams.cosineWindow = single(hann(lastLayerSize(2)) * hann(lastLayerSize(1))');
    
%     inputSize = inputSize - 1;
   
    subStride = min(inputSize ./ featrSize, 4);
    
    if strcmpi(opts.gparams.inputShape, 'square')
        opts.gparams.searchPadding = inputSize ./ targetSize - 1;
    end

    % init head --------------------------------------------
    switch lower(opts.hparams.headType)
        case 'maskconv'
            net_h = init_det_mconv(targetSize ./ subStride, opts.hparams);
        case 'baseconv'
            net_h = init_det_nconv(targetSize ./ subStride, opts.hparams);
        case 'twobranch'
            net_h = init_det_2conv(targetSize ./ subStride, opts.hparams);
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
    
    sigma = ceil(targetSize ./ subStride) * opts.tparams.motionSigmaFactor; 
    motionWindow = generate_gaussian_label(featrSize, sigma, targetSize);
    
    imageSize = round([size(img, 1), size(img, 2)] ./ resizedRatio);
    gridGenerator = dagnn.AffineGridGenerator('Ho', inputSize(2), 'Wo', inputSize(1));
   
    numScales = opts.tparams.numScales;
    scaleFactor = (-floor(numScales-1)/2):ceil((numScales-1)/2);
    scaleFactor = opts.tparams.scaleStep .^ scaleFactor;
    
    if numScales > 0
        minScaleFactor = opts.tparams.scaleStep ^ ceil(log(max(5 ./ inputSize)) / log(opts.tparams.scaleStep));
        maxScaleFactor = opts.tparams.scaleStep ^ floor(log(min(imageSize ./ targetSize([2,1]))) / log(opts.tparams.scaleStep));
    end
    
    opts.gparams.gridGenerator = gridGenerator;
    opts.gparams.resizedRatio = resizedRatio;
    opts.gparams.imageSize = imageSize;
    opts.gparams.inputSize = inputSize;
    opts.gparams.subStride = subStride;
    opts.gparams.featrSize = featrSize;
    
    opts.tparams.motionWindow = motionWindow;
    opts.tparams.scaleFactor = scaleFactor;
    opts.tparams.minSize = max(5, minScaleFactor .* targetSize);
    opts.tparams.maxSize = min(imageSize([2,1]), maxScaleFactor .* targetSize);
    
    opts.hparams.netOutIdx = state.net_h.getVarIndex('prediction');
    
    state.gparams = opts.gparams;
    state.bparams = opts.bparams;
    state.hparams = opts.hparams;
    state.tparams = opts.tparams;
    state.oparams = opts.oparams;  
    
    state.result = region;    
    state.targetRect = [region(1:2) ./ resizedRatio targetSize];
    
    state.targetScore = 1; 
    state.currFrame = 1;
    state.succIndex = 1;
end