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
    net.layers(1).block.pad = 0;
    % remove all padding
%     pLayer = find_layer_index(net, opts.removeAfterThisLayer, @arrayfun);
%     lname = {net.layers.name};
%     lname = lname(pLayer:end);
%     net.removeLayer(lname);
%     numLayers = numel(net.layers);
%     for i = 1:numLayers
%         if isa(net.layers(i).block, 'dagnn.Conv')  
% %             wsize = net.layers(i).block.size;
% %             if opts.usePad
% %                 net.layers(i).block.pad = (wsize(1)-1)/2;
% % % %             else
%                 net.layers(i).block.pad = 0;
% % %             end         
%         elseif isa(net.layers(i).block, 'dagnn.Pooling')
% % % %             wsize = net.layers(i).block.poolSize;
% % % %             if opts.usePad
% % % %                 net.layers(i).block.pad = (wsize(1)-1)/2;
% % % %             else
%                 net.layers(i).block.pad = 0;
%             end
%         end
%     end
    
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
%     net.initParams();
%     scale = 0.01;
%     value = init_weights(net.layers(1).block.size, true, scale);
%     net = assign_value(net, get_last_pindex(net), 'value', value);
%     if opts.useGpu
%         net.move('gpu');
%     end
end