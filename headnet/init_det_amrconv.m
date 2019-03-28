function net = init_det_amrconv(sz, opts)
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
    
    psize = ceil(sz / 2);
    fsize = psize * 2 + 1;
    
    regionMask = zeros(fsize([2,1]), 'single');
    intervals = ceil(fsize / opts.numRegions);
    opts.numRegions = max(ceil(fsize ./ intervals));
    for i = 1:opts.numRegions
        if i == opts.numRegions
            cfsize = fsize;
        else
            cfsize = min(i*intervals - 1 + mod(i*intervals,2), fsize);
        end
        s = ceil(size(regionMask)/2) - (cfsize([2,1])-1)/2;
        e = ceil(size(regionMask)/2) + (cfsize([2,1])-1)/2; 
        regionMask(s(1):e(1), s(2):e(2)) = regionMask(s(1):e(1), s(2):e(2)) + 1;
    end
    regionMask = regionMask / (opts.numRegions+1);
    
    block = dagnn.AMRConv('size', [fsize(2), fsize(1), lastDim, 1], ...
                          'pad', [psize(2), psize(2), psize(1), psize(1)], ...
                          'stride', [1, 1]);
    net.addLayer('detconv2', block, lastVar, 'prediction', {'detconv2f', 'detconv2b', 'detconv2m'});
    scale = 1 / sqrt(prod(fsize)*lastDim)/1e8;
    value = init_weights([fsize(2), fsize(1), lastDim, 1], true, scale);
    
    index = net.getParamIndex('detconv2f');
    net.params(index).value = value{1};
    net.params(index).learningRate = 1;
    net.params(index).weightDecay = 1;
    
    index = net.getParamIndex('detconv2b');
    net.params(index).value = value{2};
    net.params(index).learningRate = 2;
    net.params(index).weightDecay = 0;
    
    index = net.getParamIndex('detconv2m');
    net.params(index).value = regionMask;
    net.params(index).learningRate = 0;
    net.params(index).weightDecay = 0;
    
    net.addLayer('loss', dagnn.LossL2, {'prediction', 'label'}, 'loss');
    net.rebuild();
end

