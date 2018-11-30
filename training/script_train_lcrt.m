clear all; clc; close all
% addpath('dagFcn');
addpath('network');

% Prepare a CNN model for learning MDNet (windows)
% opts.initModel = 'D:/CNNModel/imagenet-vgg-m-2048.mat';
% Prepare a CNN model for learning MDNet (linux)
opts.initModel = '/media/zjjconan/Data/CNNModel/imagenet-resnet-18-dag.mat';
% -------------------------------------------------------------------------
% opts.imdbPath = '../models/imdb_vid.mat';
opts.expDir = 'data/snapshot/';
opts.outModel = '../models/res18-block2.mat';

% load imdb
% imdb = load(opts.imdbPath);
 
% network opts
opts.detNetType = 'avgfc3fc3'; 
opts.isDagNN = true;
opts.usePad = false;
opts.lossType = 'rgnet';
opts.useBNorm = false;
opts.useChnOut = false;
opts.removeAfterThisLayer = 'features_6_0_conv1';
opts.trainNet = false;

% opts.roiOpts.rpool = false;
% opts.roiOpts.align = true;
% opts.roiOpts.appendLayer = {'relu3'};
% opts.roiOpts.removeLayer = {'pool2'};
% opts.roiOpts.poolSize = [7 7];
% opts.roiOpts.transform = 1/8;

[net] = prepare_model(opts);

% % trainOpts
% trainOpts.learningRate = 0.0001 * ones(1, 100) / 128;
% trainOpts.numEpochs = numel(trainOpts.learningRate);
% trainOpts.batchSize = 1; % one video per iter
% trainOpts.epochSize = numel(imdb.images.data);
% trainOpts.derOutputs = {'all_loss', 1};
% trainOpts.gpus = [1];
% 
% batchOpts.numFrames = 8;
% batchOpts.batchPos = 32;
% batchOpts.batchNeg = 96;
% batchOpts.useGpu = trainOpts.gpus >= 1;
% batchOpts.averageImage = reshape(single([122.6769, 116.67, 104.01]), 1, 1, 3);
% batchOpts.inputSize = [75 75];
% batchOpts.padding = 4 ;
% batchOpts.outSize = round(batchOpts.inputSize * (1 + batchOpts.padding));
% batchOpts.gridGenerator = ...
%     dagnn.AffineGridGenerator('Ho', batchOpts.outSize(1), ...
%                               'Wo', batchOpts.outSize(2)); 
%                    
% % if opts.trainNet                  
% %     net = cnn_train_net(net, imdb, @(x, y) md_get_imgs_rois_frcnn(batchOpts, x, y), ...
% %         'expDir', opts.expDir, trainOpts, 'val', find(imdb.images.set == 2), ...
% %         'continue', false, 'extractStatsFn', @extractStatsMDNet, ...
% %         'plotStatistics', false);
% % %     net = cnn_train_dag(net, imdb, @(x, y) md_get_imgs_rois(batchOpts, x, y), ...
% % %         'expDir', opts.expDir, trainOpts, 'val', find(imdb.images.set == 2), ...
% % %         'continue', false,'plotStatistics', false);
% % end
%            
% net = deploy_net(net);
net.meta = [];
net = net.saveobj();
save(opts.outModel, '-struct', 'net') ;