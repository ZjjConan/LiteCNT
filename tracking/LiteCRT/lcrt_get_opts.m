function opts = lcrt_get_opts(varargin)

    gparams.verbose = false;
    gparams.useGpu = true;
    gparams.maxTargetSize = 100;
    gparams.minTargetSize = 50;
    gparams.inputShape = 'proportional';
    gparams.searchPadding = 4;
    [gparams, varargin] = vl_argparse(gparams, varargin);
    
    % backbone feature opts
    bparams.netPath = 'D:/CNNModel/imagenet-vgg-m-2048.mat';
    bparams.isDagNN = false;
    bparams.normalize = true;
    bparams.downsamplingFactor = 2;
    bparams.downsamplingMethod = 'avg';
    bparams.averageImage = single(reshape([122.6769, 116.67, 104.01], 1, 1, 3));
%     bparams.averageImage = [];
    [bparams, varargin] = vl_argparse(bparams, varargin);
    
    % online head opts
    hparams.headType = 'maskConv';
    hparams.maskSize = 5;
    hparams.alpha = 1;
    hparams.useProjection = true;
    hparams.projectInDims = 96;
    hparams.projectOuDims = 64;
    hparams.initUsePCA = true;
    [hparams, varargin] = vl_argparse(hparams, varargin);
    
    % tracking opts
    tparams.numScales = 5;
    tparams.scaleStep = 1.02;
    tparams.motionSigmaFactor = 0.7;
    tparams.scaleLr = 0.8;
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
    opts.gparams = gparams;
    opts.bparams = bparams;
    opts.hparams = hparams;
    opts.tparams = tparams;
    opts.oparams = oparams;  
end

