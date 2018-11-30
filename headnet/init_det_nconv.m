function net = init_det_nconv(sz, opts)
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
                 lastVar, 'prediction', {'detconv2f', 'detconv2b'});
    scale = 1 / sqrt(prod(filsz)*lastDim)/1e8;
    value = init_weights([filsz(2), filsz(1), lastDim, 1], true, scale);
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [1 2]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);  


    net.addLayer('loss', dagnn.LossL2, {'prediction', 'label'}, 'loss');
    net.rebuild();
    
%     if opts.useGpu
%         net.move('gpu');
%     end
end

