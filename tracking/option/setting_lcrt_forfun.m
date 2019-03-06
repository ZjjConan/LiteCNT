function opts = setting_lcrt_forfun(opts, learningRate)

 opts = lcrt_get_opts('verbose', opts.verbose, 'useGpu', opts.useGpu, ...
                         'netPath', opts.netPath, 'isDagNN', true, ...
                         'normalize', true, 'headType', 'maskconv', ...
                         'MaskSize', 5, 'downsamplingFactor', 2, ...
                         'searchPadding', 3.5, 'warmupTimes', 5, ...
                         'projectInDims', 96, 'projectOuDims', 48, ...
                         'alpha', 0.8, 'inputShape', 'square', ... 
<<<<<<< HEAD
                         'maxTargetSize', 75, 'minTargetSize', 45, ...
                         'motionSigmaFactor', 0.23, 'FIFO', true, ...
=======
                         'maxTargetSize', 70, 'minTargetSize', 45, ...
                         'motionSigmaFactor', 0.3, 'FIFO', true, ...
>>>>>>> f9f9cd232a2e7e4a8ae1ff358930a8b879abc026
                         'scaleLr', 0.45, 'scalePenalty', 0.98, 'numScales', 3, ...
                         'scaleStep', 1.03, ... tracking settings ...
                         'initMaxIters', 150, 'initMinIters', 100, ...
                         'updateMaxIters', 5, 'initLr', learningRate, ...
                         'updateLr', learningRate, 'outputSigmaFactor', 1/12);

            
    opts.state_initialize = @lcrtv2_state_initialize;
    opts.state_warmup = @lcrtv2_warmup;
    opts.initialize = @lcrtv2_initialize;
    opts.track = @lcrtv2_track;
    opts.update = @lcrt_update;
    opts.release = @lcrtv2_release;
                     
    % better than CVPR submissions
%           opts = lcrt_get_opts('verbose', opts.verbose, 'useGpu', opts.useGpu, ...
%                          'netPath', opts.netPath, 'isDagNN', true, ...
%                          'normalize', true, 'headType', 'maskconv', ...
%                          'MaskSize', 5, 'downsamplingFactor', 2, ...
%                          'searchPadding', 3.5, 'warmupTimes', 5, ...
%                          'projectInDims', 96, 'projectOuDims', 48, ...
%                          'alpha', 0.8, 'inputShape', 'proportional', ... 
%                          'maxTargetSize', 75, 'minTargetSize', 45, ...
%                          'motionSigmaFactor', 0.68, 'FIFO', true, ...
%                          'scaleLr', 0.8, 'numScales', 3, ...
%                          'scaleStep', 1.02, ... tracking settings ...
%                          'initMaxIters', 150, 'initMinIters', 100, ...
%                          'updateMaxIters', 5, 'initLr', learningRate, ...
%                          'updateLr', learningRate, 'outputSigmaFactor', 1/10);

    % 8e-6 for normal settings
%     opts.state_initialize = @lcrtnew_state_initialize;
%     opts.state_warmup = @lcrtnew_warmup;
%     opts.initialize = @lcrtnew_initialize;
%     opts.track = @lcrtnew_track;
%     opts.update = @lcrt_update;
%     opts.release = @lcrtnew_release;
   
end

