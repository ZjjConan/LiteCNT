function opts = setting_lcnt_uav123(opts, learningRate)
 
    opts = lcnt_get_opts('verbose', opts.verbose, 'useGpu', opts.useGpu, ...
                         'netPath', opts.netPath, 'isDagNN', true, ...
                         'normalize', true, 'headType', 'baseconv', ...
                         'MaskSize', 5, 'downsamplingFactor', 1, ...
                         'searchPadding', 4, 'warmupTimes', 5, ...
                         'projectInDims', 256, 'projectOuDims', 48, ...
                         'alpha', 0.8, 'inputShape', 'square', ... 
                         'maxTargetSize', 72, 'minTargetSize', 40, ...
                         'motionSigmaFactor', 0.23, 'FIFO', true, ...
                         'scaleLr', 0.8, 'numScales', 3, ...
                         'scaleStep', 1.02, ... tracking settings ...
                         'initMaxIters', 150, 'initMinIters', 100, ...
                         'updateMaxIters', 5, 'initLr', learningRate, ...
                         'updateLr', learningRate, 'outputSigmaFactor', 1/12);
     
                     

    opts.state_initialize = @lcnt_state_initialize;
    opts.state_warmup = @lcnt_warmup;
    opts.initialize = @lcnt_initialize;
    opts.track = @lcnt_track;
    opts.update = @lcnt_update;
    opts.release = @lcnt_release;   
end
