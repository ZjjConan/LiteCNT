function net = init_featrnet(varargin)
    opts.netPath = 'D:/CNNModel/imagenet-vgg-m.mat';
    opts.isDagNN = false;
    opts.usePad = false;
    opts.downsamplingFactor = 1;
    opts.downsamplingMethod = 'avg';
    [opts, varargin] = vl_argparse(opts, varargin);

    % load conv layers
    net = load(opts.netPath);
    if opts.isDagNN
        net = dagnn.DagNN.loadobj(net);
    else
        net = dagnn.DagNN.fromSimpleNN(net, 'CanonicalNames', true);
    end
    net.setLayerInputs(net.layers(1).name, {'input'});
    
    if opts.downsamplingFactor > 1
        switch opts.downsamplingMethod
            case {'max', 'avg'}
                net.addLayer('detpool1', ...
                    dagnn.Pooling('method', opts.downsamplingMethod, ...
                                  'poolSize', opts.downsamplingFactor, ...
                                  'stride', opts.downsamplingFactor, ...
                                  'pad', 0), ...
                    net.getOutputs, 'detpool1');
            case 'resize'
                
            otherwise
                error('not support');
        end
    end 
    net = sort_layers(net);
    net.rebuild();
end