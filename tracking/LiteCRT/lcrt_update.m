function state = lcrt_update(state, img)
    if(mod(state.currFrame, state.oparams.intervals)==0 )
        trainFeatrs = cat(4, state.trainFeatrs{:});
        trainLabels = cat(4, state.trainLabels{:});

        state.net_h = ...
            lcrt_finetune(state.net_h, trainFeatrs, trainLabels, ...
                          [], state.oparams, ...
                          'maxIters', state.oparams.updateMaxIters, ...
                          'learningRate', state.oparams.updateLr, ...
                          'verbose', state.oparams.verbose);
        
        if state.sparams.useTSE
            state = tse_update(img, state);
        end
    end
end

