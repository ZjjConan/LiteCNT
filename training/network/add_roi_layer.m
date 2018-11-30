function [net, lastDim] = add_roi_layer(net, lastDim, varargin)

    opts.rpool = false;
    opts.align = false;
    opts.appendLayer = {'relu3'};
    opts.removeLayer = {'pool2'};
    opts.poolSize = [3 3];
    opts.transform = 1;
    [opts, varargin] = vl_argparse(opts, varargin);
    
    lname = {net.layers.name};
    
    if ~isempty(opts.removeLayer)
        idx = find(strcmpi(lname, opts.removeLayer) == 1);
        if ~isnan(idx)
            pOut = net.layers(idx-1).outputs{1};
            net.setLayerInputs(net.layers(idx+1).name, {pOut});
        end
        
        idx = find(strcmpi(lname, opts.removeLayer) == 1);
        if ~isnan(idx)
            net.removeLayer(opts.removeLayer);
        end
    end
    
    lname = {net.layers.name};
    net.layers(8).block.dilate = [3 3];
    for i = 1:numel(opts.appendLayer)
        idx = find(strcmpi(lname, opts.appendLayer{i}) == 1);
        pOut = net.layers(idx).outputs{1};

        if opts.rpool
            block = dagnn.ROIPooling('subdivisions', opts.poolSize, 'method', 'max', ...
                'transform', opts.transform);
            net.addLayer('roipooling', block, {pOut, 'rois'}, {'roipooling'});
        elseif opts.align
            block = dagnn.ROIGridGenerator('Ho', opts.poolSize(1), 'Wo', opts.poolSize(2), ...
                                           'transform', opts.transform(i));
            net.addLayer([pOut 'roigrids'], block, {pOut, 'rois'}, {[pOut 'roigrids']});
            net.addLayer([pOut 'roifeatr'], dagnn.BilinearSampler, {pOut, [pOut 'roigrids']}, {[pOut 'roifeatr']});

            net.addLayer([pOut 'postpool'], dagnn.Pooling('poolSize', [3 3], 'stride', 2, 'method', 'max'), ...
                [pOut 'roifeatr'], [pOut 'roipooling']);
%             net.addLayer([pOut 'roinorm'], dagnn.ScaleNorm('multipliers', 100), [pOut 'roipooling'], [pOut 'roinorm']);
        end
    end
    
    if numel(opts.appendLayer) ~= 1
        net.addLayer('roifeatrconcat', dagnn.Concat, net.getOutputs, 'roifeatrconcat');
        lastDim = 96+256+512;
        block = dagnn.Conv('size', [1 1 256+96+512 512], 'stride', 1, 'pad', 0);
        value = init_weights([1 1 256+96+512 512]);
        net.addLayer('roifeatr', block, 'roifeatrconcat', 'roifeatr', {'roifeatrf', 'roifeatrb'});
        net = assign_value(net, get_last_pindex(net), 'value', value);
        net = assign_value(net, get_last_pindex(net), 'learningRate', [10 20]);
        net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);  
        net.addLayer('roifeatrx', dagnn.ReLU, 'roifeatr', 'roifeatrx');
        lastDim = 512;
    end
end

