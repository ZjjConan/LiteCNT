function [net, info] = init_net_fc3fc1fc1(net, lastDim, numBranches, useBNorm)
    % Block 4
    block = dagnn.Conv('size', [3 3 lastDim 512], 'stride', 1, 'pad', 0);
    value = init_weights([3 3 lastDim 512]);
      
    net.addLayer('detconv1', block, net.getOutputs, 'detconv1', {'detconv1f', 'detconv1b'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [10 20]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);  
    net.addLayer('detconv1x', dagnn.ReLU, 'detconv1', 'detconv1x');
    
    if useBNorm
        net = BNorm(net, 'detreg1', 'detconv1', 512);
        net.setLayerInputs('detconv1x', {'detreg1'});
    else
        net.addLayer('detdrop1', dagnn.DropOut, 'detconv1x', 'detreg1');
    end

    % Block 4
    block = dagnn.Conv('size', [1 1 512 512], 'stride', 1, 'pad', 0);
    value = init_weights([1 1 512 512]);
    
    net.addLayer('detconv2', block, net.getOutputs, 'detconv2', {'detconv2f', 'detconv2b'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [10 20]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);
    
    net.addLayer('detconv2x', dagnn.ReLU, 'detconv2', 'detconv2x');
    
    if useBNorm
        net = BNorm(net, 'detreg2', 'detconv2', 512);
        net.setLayerInputs('detconv2x', {'detreg2'});
    else
        net.addLayer('detdrop2', dagnn.DropOut, 'detconv2x', 'detreg2');
    end 

    % Block 6
    block = dagnn.Conv('size', [1 1 512 2*numBranches], 'stride', 1, 'pad', 0);
    value = init_weights([1 1 512  2*numBranches]);
    
    net.addLayer('detconv3', block, net.getOutputs, 'prediction', {'detconv3f', 'detconv3fb'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [10 20]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);

    info.detNetType = 'fc3-512-fc1-512-fc1-2';
end