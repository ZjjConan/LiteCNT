function opts = lcrt_get_opts(varargin)

    gparams.verbose = false;
    gparams.useGpu = true;
    gparams.maxTargetSize = 100;
    gparams.minTargetSize = 50;
    gparams.inputShape = 'proportional';
    gparams.searchPadding = 4;
    gparams.warmupTimes = 5;
    gparams.useDataAugmentation = true;
    [gparams, varargin] = vl_argparse(gparams, varargin);
    
    % backbone feature opts
    bparams.netPath = '';
    bparams.isDagNN = false;
    bparams.normalize = true;
    bparams.downsamplingFactor = 2;
    bparams.downsamplingMethod = 'avg';
    bparams.averageImage = single(reshape([122.6769, 116.67, 104.01], 1, 1, 3));
    [bparams, varargin] = vl_argparse(bparams, varargin);
    
    % augmentation opts
    aparams = struct();
    aparams(1).type = 'fliplr';
    aparams(1).param = [];
    aparams(2).type = 'rot';
    aparams(2).param = {5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60};
    aparams(3).type = 'blur';
    aparams(3).param = {[2, 0.2], [0.2, 2], [3, 1], [1, 3], [2, 2]};
    aparams(4).type = 'shift';
    aparams(4).param = {[8, 8], [-8, 8], [8, -8], [-8, -8]};
    aparams(5).type = 'dropout';
    aparams(5).param = {1, 2, 3, 4, 5, 6, 7};
    [aparams, varargin] = vl_argparse(aparams, varargin);
    
    % online head opts
    hparams.headType = 'maskConv';
    hparams.kernelRatio = 1;
    hparams.maskSize = 5;
    hparams.alpha = 1;
    hparams.useProjection = true;
    hparams.projectInDims = 96;
    hparams.projectOuDims = 64;
    hparams.initUsePCA = true;
    [hparams, varargin] = vl_argparse(hparams, varargin);
    
    % tracking opts
    % target state estimation params
    sparams.useTSE = false;
    sparams.useWEstimator = true;
    sparams.useHEstimator = true;
    sparams.TSESigmaFactor = 1/12;
    sparams.TSELearningRate = 0.025;
    sparams.TSENumScaleFilters = 7;
    sparams.TSENumInterpScales = 13;
    sparams.TSEScaleFactor = 1.05;
    sparams.TSEMaxTargetSize = 31;
    sparams.TSEMinTargetSize = 9;
    sparams.TSELambda = 1e-2;
    
    [sparams, varargin] = vl_argparse(sparams, varargin);
    
%     tparams.use_scale_filter = true;         % Use the fDSST scale filter or not (for speed)
%     tparams.scale_sigma_factor = 1/16;       % Scale label function sigma
%     tparams.scale_learning_rate = 0.025;		% Scale filter learning rate
%     % params.number_of_scales_filter = 5;    % Number of scales
%     tparams.number_of_scales_filter = 17; 
%     tparams.number_of_interp_scales = 33;    % Number of interpolated scales
%     tparams.scale_model_factor = 1.0;        % Scaling of the scale model
%     tparams.scale_step_filter = 1.02;        % The scale factor for the scale filter
%     tparams.scale_model_max_area = 32*16;    % Maximume area for the scale sample patch
%     tparams.scale_feature = 'HOG4';          % Features for the scale filter (only HOG4 supported)
%     tparams.s_num_compressed_dim = 'MAX';    % Number of compressed feature dimensions in the scale filter
%     tparams.lambda = 1e-2;					% Scale filter regularization
%     tparams.do_poly_interp = true;           % Do 2nd order polynomial interpolation to obtain more accurate scale
%     tparams.useBBR = true;
%     tparams.BBRScaleFactor = 1.05;
%     tparams.BBRInitNums = 1000;
%     tparams.BBRUpdateNums = 10;
%     tparams.BBRFeatrSize = [3 3];
    
    % other
    tparams.numScales = 5;
    tparams.scaleStep = 1.02;
    tparams.motionSigmaFactor = 0.7;
    tparams.scaleLr = 0.8;
    tparams.scalePenalty = 0.8;
    [tparams, varargin] = vl_argparse(tparams, varargin);
    
    % optimization opts
    oparams.outputSigmaFactor = 1/10;
    oparams.initLr = 1e-5;
    oparams.initMaxIters = 400;
    oparams.initMinIters = 100;
    oparams.updateLr = 1e-5;
    oparams.updateMaxIters = 2;
    oparams.numSamples = 2;
    oparams.intervals = 5;
    oparams.FIFO = true;
    oparams.verbose = gparams.verbose ;
    [oparams, varargin] = vl_argparse(oparams, varargin);
    
    % sample remove opts  
    opts.aparams = aparams; 
    opts.sparams = sparams;
    opts.gparams = gparams;
    opts.bparams = bparams;
    opts.hparams = hparams;
    opts.tparams = tparams;
    opts.oparams = oparams;  
end

