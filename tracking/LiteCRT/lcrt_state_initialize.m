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
   
    subStride = inputSize ./ featrSize;
    
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
%     motionWindow =  single(hann(featrSize(2)) * hann(featrSize(1))');
    
    imageSize = round([size(img, 2), size(img, 1)] ./ resizedRatio);
    gridGenerator = dagnn.AffineGridGenerator('Ho', inputSize(2), 'Wo', inputSize(1));
   
    numScales = opts.tparams.numScales;
    scaleFactor = (-floor(numScales-1)/2):ceil((numScales-1)/2);
    scaleFactor = opts.tparams.scaleStep .^ scaleFactor;
%     scaleFactor = [scaleFactor; scaleFactor];
%     scaleFactor = [[1; scaleFactor(1, 1)], ...
%                   [scaleFactor(1, 1); 1], ...
%                   scaleFactor(:, 2), ...
%                   [scaleFactor(1, 3); 1], ...
%                   [1; scaleFactor(1, 3)]];
    if numScales > 0
        minScaleFactor = opts.tparams.scaleStep ^ ceil(log(max(5 ./ inputSize)) / log(opts.tparams.scaleStep));
        maxScaleFactor = opts.tparams.scaleStep ^ floor(log(min(imageSize ./ targetSize)) / log(opts.tparams.scaleStep));
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
    opts.tparams.minSize = max(5, minScaleFactor .* targetSize);
    opts.tparams.maxSize = min(imageSize, maxScaleFactor .* targetSize);
    
    opts.hparams.netOutIdx = state.net_h.getVarIndex('prediction');
    
    state.aparams = aparams;
    state.gparams = opts.gparams;
    state.bparams = opts.bparams;
    state.hparams = opts.hparams;
    state.tparams = opts.tparams;
    state.oparams = opts.oparams;  
    
    state.result = region;    
    state.targetRect = [region(1:2) ./ resizedRatio targetSize];
    state.scaledTargetSize = targetSize;
    
    state.targetScore = 1; 
    state.currFrame = 1;
    state.succIndex = 1;
end