function opts = lcnt_get_opts(varargin)

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
    
    % online head opts
    hparams.headType = 'amrconv';
    hparams.kernelRatio = 1;
    hparams.maskSize = 5;
    hparams.alpha = 1;
    hparams.useProjection = true;
    hparams.projectInDims = 96;
    hparams.projectOuDims = 64;
    hparams.numRegions = 2;
    hparams.initUsePCA = true;
    [hparams, varargin] = vl_argparse(hparams, varargin);
            
    % data augment opts
    aparams = struct();
    aparams(end).type = 'fliplr';
    aparams(end).param = [];
    aparams(end+1).type = 'rot';
    aparams(end).param = {5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60};
    aparams(end+1).type = 'blur';
    aparams(end).param = {[2, 0.2], [0.2, 2], [3, 1], [1, 3], [2, 2]};
    aparams(end+1).type = 'shift';
    aparams(end).param = {[8, 8], [-8, 8], [8, -8], [-8, -8]};
    aparams(end+1).type = 'dropout';
    aparams(end).param = {1, 2, 3, 4, 5, 6, 7};
    [aparams, varargin] = vl_argparse(aparams, varargin);
    
    % other
    tparams.numScales = 5;
    tparams.scaleStep = 1.02;
    tparams.motionSigmaFactor = 0.7;
    tparams.scaleLr = 0.8;
    tparams.scalePenalty = 0.98;
    [tparams, varargin] = vl_argparse(tparams, varargin);
    
    % optimization opts
    oparams.outputSigmaFactor = 1/10;
    oparams.hogomSigma = true;
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
    opts.gparams = gparams;
    opts.bparams = bparams;
    opts.hparams = hparams;
    opts.tparams = tparams;
    opts.oparams = oparams;  
end

