function [net, info] = prepare_model(varargin)
    % conv1-3 layers from VGG-M network pretrained on ImageNet
    opts.initModel = 'D:/CNNModel/imagenet-vgg-m.mat';
    opts.isDagNN = false;
    opts.detNetType = 'fc3fc1fc1';
    opts.numBranches = 1;
    opts.removeAfterThisLayer = 'conv4';
    opts.usePad = false;
    opts.lossType = 'mdnet';
    opts.useBNorm = false;
    opts.useChnOut = true;
    
    opts.roiOpts.rpool = false;
    opts.roiOpts.align = false;
    opts.roiOpts.layer = {'relu3'};
    opts.roiOpts.poolSize = [3 3];
    opts.roiOpts.transform = 1;
    [opts, varargin] = vl_argparse(opts, varargin);

    % load conv layers
    net = load(opts.initModel);
    if opts.isDagNN
        net = dagnn.DagNN.loadobj(net);
    else
        net = dagnn.DagNN.fromSimpleNN(net, 'CanonicalNames', true);
    end
    net.setLayerInputs(net.layers(1).name, {'input'});
    
    % remove all padding
    pLayer = find_layer_index(net, opts.removeAfterThisLayer, @arrayfun);
    lname = {net.layers.name};
    lname = lname(pLayer:end);
    net.removeLayer(lname);
    numLayers = numel(net.layers);
   
    for i = 1:numLayers
        if isa(net.layers(i).block, 'dagnn.Conv')
%             net.params(net.getParamIndex([net.layers(i).name 'f'])).learningRate = 1;
%             net.params(net.getParamIndex([net.layers(i).name 'b'])).learningRate = 2;
            
            wsize = net.layers(i).block.size;
            
            if opts.usePad
                net.layers(i).block.pad = (wsize(1)-1)/2;
            end
%             
%             if opts.useBNorm
%                 net = BNorm(net, [net.layers(i).name 'bn'], net.layers(i).outputs{1}, wsize(end));
%                 net.setLayerInputs(net.layers(i+1).name, {[net.layers(i).name 'bn']});
%             end
%             
%             lastDim = wsize(end);
%             
        elseif isa(net.layers(i).block, 'dagnn.Pooling')
            wsize = net.layers(i).block.poolSize;
            net.layers(i).block.poolSize = [2 2];
            net.layers(i).block.pad = 0;
%             if opts.usePad
%                 net.layers(i).block.pad = (wsize(1)-1)/2;
%             end
        end
    end
    lastDim = 32;
    if opts.roiOpts.rpool | opts.roiOpts.align
        [net, lastDim] = add_roi_layer(net, lastDim, opts.roiOpts);
    end
    
    
    
%     if opts.useChnOut
%         for i = 1:numLayers
%             if isa(net.layers(i).block, 'dagnn.Conv')
%                 if i == 1, continue; end
%                 lname = [net.layers(i-1).name 'cout'];
%                 net.addLayer(lname, dagnn.ChnOut, net.layers(i-1).outputs{1}, lname);
%                 if i == numLayers, break; end
%                 net.setLayerInputs(net.layers(i).name, {lname});
%             end
%         end
%     end
    
%     for i = numel(net.layers):-1:1
%         if isa(net.layers(i).block, 'dagnn.Conv')
%             lastDim = net.layers(i).block.size(end); 
%             break;
%         end
%     end
    
%     switch opts.detNetType
%         case 'fc3fc1fc1', init_net_fcn = @init_net_fc3fc1fc1;
%         case 'fc2fc1fc1', init_net_fcn = @init_net_fc2fc1fc1;
%         case 'avgfc3fc3', init_net_fcn = @init_net_avgfc3fc3;
%         otherwise, error('network type is not supported in current implementation');
%     end
% 
%     
%     [net, info] = init_net_fcn(net, lastDim, opts.numBranches, opts.useBNorm);
%     
%     switch (opts.lossType)
%         case 'rgnet'
%             pOut = net.layers(end).name;
%             net.addLayer('loss', dagnn.LossL2, {'prediction', 'label'}, 'loss');
%         case 'mdnet'
%            
%             pOut = net.layers(end).name;
%          
%             net.addLayer('all_loss', dagnn.SoftMaxKLoss('numBranches', opts.numBranches), ...
%             {'prediction', 'k', 'label'}, 'all_loss');
%         
%             net.addLayer('pos_err', dagnn.MDPosError('numBranches', opts.numBranches), ...
%             {'prediction', 'k', 'label'}, 'pos_err');
% 
%             net.addLayer('neg_err', dagnn.MDNegError('numBranches', opts.numBranches), ...
%             {'prediction', 'k', 'label'}, 'neg_err');
% 
%             net.addLayer('all_err', dagnn.MDError('numBranches', opts.numBranches), ...
%             {'prediction', 'k', 'label'}, 'all_err');
%         case 'mdnet_ins'
%             pOut = net.layers(end).name;
%             net.addLayer('all_loss', dagnn.SoftMaxKLoss('numBranches', opts.numBranches), ...
%             {'prediction', 'k', 'label'}, 'all_loss');
%         
%             net.addLayer('ins_loss', dagnn.InsSoftMaxLoss, {'prediction', 'ins_label'}, 'ins_loss'); 
%         case 'sdnet'
%             pOut = net.layers(end).name;
%             net.addLayer('loss', dagnn.Loss, {'prediction', 'label'}, 'loss');
%             net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prediction', 'label'}, 'error');
%         case 'clynet'
%             pOut = net.layers(end).name;
%             net.addLayer('loss', dagnn.Loss, {'prediction', 'label'}, 'loss');
%             net.addLayer('all_err', dagnn.Loss('loss', 'classerror'), ...
%                  {'prediction','label'}, 'all_err') ;
%             net.addLayer('pos_err', dagnn.ClyPosError, {'prediction','label'}, 'pos_err') ;
%             net.addLayer('neg_err', dagnn.ClyNegError, {'prediction','label'}, 'neg_err') ;
% %             net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
% %                  'opts', {'topK',5}), ...
% %                  {'prediction','label'}, 'top5err') ;
%     end
%     
%     info.trainLossType = opts.lossType;
%     net.setLayerOutputs(pOut, {'prediction'});
    net = sort_layers(net);
    net.rebuild();
end

function net = BNorm(net, lName, inName, chns)
    block = dagnn.BatchNorm('numChannels', chns, 'epsilon', 1e-5);
    weights = block.initParams();
    net.addLayer(lName, block, inName, lName, {[lName 'f'], [lName 'b'], [lName 'm']});
    net.params(end-2).value = weights{1};
    net.params(end-1).value = weights{2};
    net.params(end  ).value = weights{3}; 
end