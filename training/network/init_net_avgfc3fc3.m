function [net, info] = init_net_avgfc3fc3(net, lastDim, numBranches, useBNorm)
%     net.layers(1).block.stride = 2;
    net.addLayer('detpool1', dagnn.Pooling('method', 'avg', 'poolSize', 3, 'stride', 2, 'pad', 1), net.getOutputs, 'detpool1');
    
%     net.addLayer('detpool1', dagnn.SplitPlane('stride', 2), net.getOutputs, 'detpool1');

    % Block 4
    block = dagnn.Conv('size', [3 3 lastDim 64], 'stride', 1, 'pad', 1);
    value = init_weights([3 3 lastDim 64], true, 1/sqrt(3*3*lastDim)/1e8);
%         value = init_weights([3 3 lastDim 64], true, 1e-6);
%     value = block.initParams();
      
    net.addLayer('detconv1', block, net.getOutputs, 'detconv1', {'detconv1f', 'detconv1b'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [1 2]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 1]);  
    net.addLayer('detconv1x', dagnn.ReLU, 'detconv1', 'detconv1x');
    
    
    block = dagnn.Conv('size', [1 1 64 1], 'stride', 1, 'pad', 0);
    value = init_weights([1 1 64 1], true, 1/sqrt(1*1*64)/1e8);
      
    net.addLayer('detconv2', block, net.getOutputs, 'detconv2', {'detconv2f', 'detconv2b'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [1 2]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 1]);  
    net.addLayer('detconv2x', dagnn.ReLU, 'detconv2', 'detconv2x');
    
    
%     if useBNorm
%         net = BNorm(net, 'detreg1', 'detconv1', 512);
%         net.setLayerInputs('detconv1x', {'detreg1'});
%     else
%         net.addLayer('detdrop1', dagnn.DropOut, 'detconv1x', 'detreg1');
%     end

    % Block 4
%     block = dagnn.Conv('size', [1 1 64 1], 'stride', 1, 'pad', 0);
%     value = init_weights([1 1 64 1], true, 1*1/1e8);
% %     
%     net.addLayer('detconv3', block, net.getOutputs, 'detconv3', {'detconv3f', 'detconv3b'});
%     net = assign_value(net, get_last_pindex(net), 'value', value);
%     net = assign_value(net, get_last_pindex(net), 'learningRate', [1 2]);
%     net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);
    
%     net.addLayer('detconv2x', dagnn.ReLU, 'detconv2', 'detconv2x');
    
%     if useBNorm
%         net = BNorm(net, 'detreg2', 'detconv2', 512);
%         net.setLayerInputs('detconv2x', {'detreg2'});
%     else
%         net.addLayer('detdrop2', dagnn.DropOut, 'detconv2x', 'detreg2');
%     end 

    info.detNetType = 'fc3-16-fc3-1';
end