function state = accumulate_gradients(net, state, params, batchSize)

    for p = 1:numel(net.params)      
        parDer = net.params(p).der ;
        if isempty(parDer), continue; end

        thisDecay = params.weightDecay * net.params(p).weightDecay ;
        thisLR = params.learningRate * net.params(p).learningRate ;

        if thisLR>0 || thisDecay>0
            % Normalize gradient and incorporate weight decay.
            parDer = vl_taccum(1/batchSize, parDer, thisDecay, net.params(p).value) ;

            if isempty(params.solver)
                state.solverState{p} = vl_taccum(...
                    params.momentum, state.solverState{p}, ...
                    -1, parDer) ;

                delta = state.solverState{p} ;

                net.params(p).value = vl_taccum(1,  net.params(p).value, thisLR, delta) ;

            else % call solver function to update weights
                [net.params(p).value, state.solverState{p}] = ...
                    params.solver(net.params(p).value, state.solverState{p}, ...
                    parDer, params.solverOpts, thisLR) ;
            end
        end
    end
end