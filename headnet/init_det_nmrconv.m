function net = init_det_nmrconv(sz, opts)
    net = dagnn.DagNN();
    
    if opts.useProjection
        net.addLayer('detconv1', dagnn.Conv('size', [1, 1, opts.projectInDims, opts.projectOuDims], ...
                     'hasBias', true, 'pad', 0, 'stride', [1,1]), ...
                     'input', 'detconv1', {'detconv1f', 'detconv1b'});
        scale = 1 / sqrt(1*1*opts.projectInDims)/1e8;
        value = init_weights([1, 1, opts.projectInDims, opts.projectOuDims], true, scale);
        net = assign_value(net, get_last_pindex(net), 'value', value);
        net = assign_value(net, get_last_pindex(net), 'learningRate', [1 2]);
        net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);  
        lastVar = 'detconv1';
        lastDim = opts.projectOuDims;
    else
        lastVar = 'input';
        lastDim = opts.projectInDims;
    end
    
    padsz = ceil(sz / 2);
    filsz = padsz * 2 + 1;  
    net.addLayer('detconv2', dagnn.Conv('size', [filsz(2), filsz(1), lastDim, 1], ...
                 'hasBias', true, 'pad', [padsz(2), padsz(2), padsz(1), padsz(1)], 'stride', [1, 1]), ...
                 lastVar, 'region_out1', {'detconv2f', 'detconv2b'});
    scale = 1 / sqrt(prod(filsz)*lastDim)/1e8;
    value = init_weights([filsz(2), filsz(1), lastDim, 1], true, scale);
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [1 2]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);  
    
    
    intervals = ceil(filsz / opts.numMultiLevels);
    
    filsz_2 = min(intervals - 1 + mod(intervals,2), filsz);
    padsz_2 = (filsz_2 - 1) / 2;
    
   
    net.addLayer('detconv3', dagnn.Conv('size', [filsz_2(2), filsz_2(1), lastDim, 1], ...
                 'hasBias', true, 'pad', [padsz_2(2), padsz_2(2), padsz_2(1), padsz_2(1)], 'stride', [1, 1]), ...
                 lastVar, 'region_out2', {'detconv3f', 'detconv3b'});
    scale = 1 / sqrt(prod(filsz_2)*lastDim)/1e8;
    value = init_weights([filsz_2(2), filsz_2(1), lastDim, 1], true, scale);
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [1 2]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);  
    
    net.addLayer('detsum', dagnn.Sum, {'region_out1', 'region_out2'}, 'prediction');
    
    net.addLayer('loss', dagnn.LossL2, {'prediction', 'label'}, 'loss');
    net.rebuild();
end

