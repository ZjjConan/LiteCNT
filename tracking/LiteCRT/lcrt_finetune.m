function [net, state] = lcrt_finetune(net, trainFeatrs, trainLabels, state, varargin)

    opts.verbose = false;
    opts.useGpu = true;
    opts.conserveMemory = true ;
    opts.sync = true ;

    opts.maxIters = 30;
    opts.minIters = 30;
    opts.learningRate = 0.0001;
    opts.weightDecay = 5e-4 ;
    opts.momentum = 0.9 ;
    opts.startFrame = false;

    opts.backPropDepth = inf;

    opts.solver = [];
    opts.solverOpts = opts.solver();
    opts.nesterovUpdate = false;
    [opts, varargin] = vl_argparse(opts, varargin) ;
    
    opts.solver = @solver.adam;
    opts.solverOpts = opts.solver();
    
    % ---------------------------------------------------------------------
    %                                                Network initialization
    % ---------------------------------------------------------------------
    state = [];
    if isempty(state) || isempty(state.solverState)
        state.solverState = cell(1, numel(net.params)) ;
        state.solverState(:) = {0} ;
    else
        for s = 1:numel(state.solverState)
            if isfield(state.solverState{s}, 't')
                state.solverState{s}.t = 0;
            end
        end
    end


    % -----------
    % Initilizing
    if opts.useGpu
        one = gpuArray(single(1)) ;
        trainFeatrs = gpuArray(trainFeatrs);
        trainLabels = gpuArray(trainLabels);
    else
        one = single(1) ;
    end

    numData = size(trainFeatrs, 4);
    perm = prepare_data_list(numData, 1, opts.maxIters);
  
%     if ~opts.startFrame     
%         perm = zeros(1, opts.maxIters);
%         perm(1:2:end) = 1;
%         perm(2:2:end) = numData;
%     end
    % objective fuction
    obj = zeros(1, opts.maxIters);
    pPred = net.getVarIndex('prediction');
    pLoss = net.getVarIndex('loss');
    pLabl = net.getVarIndex('label');
    net.vars(pPred).precious = 1;
    net.vars(pLoss).precious = 1;
    net.vars(pLabl).precious = 1;
    % training on training set
    net.mode = 'normal';
    net.reset();
    for t = 1:opts.maxIters 
        if opts.verbose
            fprintf('\ttraining batch %3d of %3d ... ', t, opts.maxIters) ;
        end
        excuTime = tic ;
        findex = perm(t);
        if opts.startFrame       
            lindex = 1;
        else
            lindex = perm(t); 
        end
        batchInputs = cat(2, {'input', trainFeatrs(:,:,:,findex)}, {'label', trainLabels(:,:,:,lindex)});
        net.eval(batchInputs, {'loss', one});
        net.vars(pLabl).value = [];
        state = accumulate_gradients(net, state, opts, numel(lindex));
        obj(t) = gather(net.vars(pLoss).value) / numel(lindex);
        
        excuTime = toc(excuTime);

        if opts.verbose
            fprintf('network training batch %3d of %3d ---- obj %.3f, %.3fs\n', ...
                t, opts.maxIters, obj(t), excuTime) ;
        end  
        
        if obj(t) < 0.02 && t > opts.minIters
            break;
        end
    end 
    
    net.mode = 'test';
end


function dataPerm = prepare_data_list(nums, batch, maxIters)
    dataPerm = randperm(nums);
    dataPerm = repmat(dataPerm, 1, ceil(maxIters/nums));
    dataPerm = dataPerm(1:batch:maxIters);
end