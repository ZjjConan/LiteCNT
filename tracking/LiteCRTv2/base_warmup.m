function state = base_warmup(state)

    if state.gparams.warmupTimes > 0
        % store tracking params
        targetRect = state.targetRect;
        scaledTargetSize = state.scaledTargetSize;
        initMinIters = state.oparams.initMinIters;
        initMaxIters = state.oparams.initMaxIters;
        initLr = state.oparams.initLr;
        
        % set a very lite params for warmup
        state.targetRect = [20 20 20 20];
        state.scaledTargetSize = [20 20];
        state.oparams.initMinIters = 1;
        state.oparams.initMaxIters = 1;
        state.oparams.initLr = 0;
        
        for i = 1:state.gparams.warmupTimes
            img = randn([200 200 3], 'single');
    
            if state.gparams.useGpu
                img = gpuArray(img);
            end
            state = base_initialize(state, img);    
        end
        
        % get tracking params backup
        state.targetRect = targetRect;
        state.scaledTargetSize = scaledTargetSize;
        state.oparams.initMinIters = initMinIters;
        state.oparams.initMaxIters = initMaxIters;
        state.oparams.initLr = initLr;
    end
    state.gparams.warmup = false;
end

