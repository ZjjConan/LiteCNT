function state = lcrtup_state_initialize(img, region, opts)
    % init network for feature extraction    
    net_f = init_featrnet(opts);
    
    % contraint the target size for efficient tracking
    if max(region(3:4)) > opts.maxTargetSize
        resizedRatio = max(region(3:4)) ./ opts.maxTargetSize;
    elseif max(region(3:4)) < opts.minTargetSize
        resizedRatio = max(region(3:4)) ./ opts.minTargetSize;
    else
        resizedRatio = 1;
    end
    targetSize = round(region(3:4) ./ resizedRatio); 
    
    % determine the output size and subsampling factor
    switch opts.inputShape
        case 'square'
            opts.inputSize = round(repmat(sqrt(prod(targetSize * (1 + opts.padding))), 1, 2));
        case 'proportional'
            opts.inputSize = round(targetSize * (1 + opts.padding));
    end
    
    opts.cosineWindow = [];
    featr = lcrt_extract_feature(net_f, randn([opts.inputSize([2,1]) 3], 'single'), opts);
    featrSize = [size(featr, 2), size(featr, 1)];
    if mod(featrSize(1), 2) == 0, featrSize(1) = featrSize(1) + 1; end
    if mod(featrSize(2), 2) == 0, featrSize(2) = featrSize(2) + 1; end
    opts.cosineWindow = single(hann(featrSize(2)) * hann(featrSize(1))');
    
    opts.inputSize = get_input_size(net_f, featrSize);
    opts.subStride = opts.inputSize ./ featrSize;
    if strcmpi(opts.inputShape, 'square')
        opts.padding = opts.inputSize ./ targetSize - 1;
    end

    % init det --------------------------------------------
    if opts.useMaskConv
        net_d = init_det_mconv(targetSize ./ opts.subStride, opts);
    else
        net_d = init_det_nconv(targetSize ./ opts.subStride, opts);
    end
    % ---------------------------------------------------------
     
    state.paramSize = get_model_size(net_f);
    state.paramSize = state.paramSize + get_model_size(net_d);
    
    net_f.addLayer('window', dagnn.CosWindow('window', opts.cosineWindow), net_f.getOutputs{1}, {'window'});
    if opts.featrNormalize
        net_f.addLayer('featrn', dagnn.FeatrNorm, net_f.getOutputs{1}, 'featrn');
        outputs = 'featrn';
    else
        outputs = 'window';
    end
    
    net_f = net_f.saveobj();
    net_d = net_d.saveobj();
    net_d.layers = [net_f.layers, net_d.layers];
    net_d.params = [net_f.params, net_d.params];
    clear net_f;
    state.net_d = dagnn.DagNN.loadobj(net_d);
    state.net_d.setLayerInputs('detconv1', {outputs});
    state.net_d.params(1).learningRate = 10;
    state.net_d.params(2).learningRate = 20;
    if opts.useGpu
        state.net_d.move('gpu');
        opts.averageImage = gpuArray(opts.averageImage);
    end
    
    sigma = ceil(targetSize ./ opts.subStride) * opts.motionSigmaFactor; 
    state.motionWindow = generate_gaussian_label(featrSize, sigma, targetSize);
    
    opts.imageSize = round([size(img, 1), size(img, 2)] ./ resizedRatio);
    opts.gridGenerator = dagnn.AffineGridGenerator('Ho', opts.inputSize(2), 'Wo', opts.inputSize(1));
   
    numScales = opts.numScales;
    scaleFactor = (-floor(numScales-1)/2):ceil((numScales-1)/2);
    scaleFactor = opts.scaleStep .^ scaleFactor;
    
    if numScales > 0
        minScaleFactor = opts.scaleStep ^ ceil(log(max(5 ./ opts.inputSize)) / log(opts.scaleStep));
        maxScaleFactor = opts.scaleStep ^ floor(log(min(opts.imageSize ./ targetSize([2,1]))) / log(opts.scaleStep));
    end
    
    state.opts = opts;
    state.resizedRatio = resizedRatio;
    state.featrSize = featrSize;
    state.scaleFactor = scaleFactor;
    state.minSize = max(5, minScaleFactor .* targetSize);
    state.maxSize = min(opts.imageSize([2,1]), maxScaleFactor .* targetSize);
    
    state.netOutIdx = state.net_d.getVarIndex('prediction');
    state.pcaOutIdx = state.net_d.getVarIndex(outputs);
    state.net_d.vars(state.netOutIdx).precious = 1;
    state.net_d.vars(state.pcaOutIdx).precious = 1;
    state.result = region;    
    state.targetRect = [region(1:2) ./ resizedRatio targetSize];
    
    state.targetScore = 1; 
    state.currFrame = 1;
    state.succIndex = 1;
end