function net = init_det_mconv(sz, opts)
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
    
%     sigma = ceil(fsize .* 1/8) ;
    multiLevelMask = zeros(fsize([2,1]), 'single');
    intervals = floor(fsize / 5);

    for i = 1:5
        if i == 5
            cfsize = fsize;
        else
            cfsize = min(i*intervals - 1 + mod(i*intervals,2), fsize);
        end
        s = ceil(size(multiLevelMask)/2) - (cfsize([2,1])-1)/2;
        e = ceil(size(multiLevelMask)/2) + (cfsize([2,1])-1)/2; 
        multiLevelMask(s(1):e(1), s(2):e(2)) = multiLevelMask(s(1):e(1), s(2):e(2)) + 1;
    end
    multiLevelMask = multiLevelMask / max(multiLevelMask(:));
%     mask = (1 - opts.alpha) * ones(fsize([2,1]), 'single');
%     mask = generate_gaussian_label(fsize, sigma);
%     if opts.alpha ~= 1
% %         cfsize = ceil(fsize * 0.3);
% %         cpsize = max(1, ceil(psize * 0.3));
% %         cfsize =  cpsize * 2 + 1;
%         cfsize = fsize;
%         if max(cfsize) > opts.maskSize
%             cfsize = ceil(cfsize .* opts.maskSize / max(cfsize));
%         end
%         s = ceil(size(mask)/2) - (cfsize([2,1])-1)/2;
%         e = ceil(size(mask)/2) + (cfsize([2,1])-1)/2; 
%         mask(s(1):e(1), s(2):e(2)) = opts.alpha;
%     end

    
    block = dagnn.MaskConv('size', [fsize(2), fsize(1), lastDim, 1], ...
                           'pad', [psize(2), psize(2), psize(1), psize(1)], ...
                           'stride', [1, 1], ...
                           'numMultiLevels', 5);
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
    net.params(index).value = multiLevelMask;
    net.params(index).learningRate = 0;
    net.params(index).weightDecay = 0;
    
    net.addLayer('loss', dagnn.LossL2, {'prediction', 'label'}, 'loss');
    net.rebuild();
end

