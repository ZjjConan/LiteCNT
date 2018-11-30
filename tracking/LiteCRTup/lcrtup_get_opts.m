function opts = lcrtup_get_opts(varargin)

    opts.verbose = false;
    opts.useGpu = true;
    
    % network opts
    opts.initModel = 'D:/CNNModel/imagenet-vgg-m-2048.mat';
    opts.isDagNN = false;
    opts.usePad = true;
    opts.useMaskConv = true;
    opts.removeAfterThisLayer = 'pool1';
    opts.downsamplingFactor = 2;
    
    % feature opts
    opts.featrDims = 96;
    opts.PCADims = 64;
    opts.featrNormalize = true;
    
    % center-bias mask opts
    opts.alpha = 1;
    opts.padding = 4;
    opts.inputShape = 'proportional';
    
    % normalized target size
    opts.maxTargetSize = 100;
    opts.minTargetSize = 50;
    
    % scale for tracking
    opts.numScales = 5;
    opts.scaleStep = 1.02;
    
    % learning parameters
    opts.outputSigmaFactor = 1/10;
    opts.motionSigmaFactor = 0.7;
    
    % initialize learning parameters
    opts.initLr = 1e-5;
    opts.initMaxIters = 400;
    opts.initMinIters = 100;
    
    % update learning parameters
    opts.updateLr = 1e-5;
    opts.updateMaxIters = 2;
    
    % scale learning rate
    opts.scaleLr = 0.8;
    
    % update samples and intervals
    opts.numSamples = 2;
    opts.intervals = 5;
    
    % sample remove opts
    opts.FIFO = true;

    opts.averageImage = single(reshape([122.6769, 116.67, 104.01], 1, 1, 3));
    
    [opts, varargin] = vl_argparse(opts, varargin);
end

