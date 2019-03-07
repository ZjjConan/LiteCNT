clc; clear all; close all;
warning off;

opts.videoPath = '/media/zjjconan/Data/Dataset/Video/Temple-color-128/';
opts.savePath = '/media/zjjconan/Experiments/tracker_zoo/Evaluation/results/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('otb_distribute.mat');

<<<<<<< HEAD
for learningRate = [9e-6]
%     learningRate = 1e-5;
    opts.trackerName = ['lasot_LiteCRTv3-' num2str(learningRate)];
%     opts.trackerName = 'LiteCRTv2';
=======
for learningRate = [7e-6]
%     learningRate = 8e-6;
    opts.trackerName = ['tpl_LiteCRTv3-TSE2-' num2str(learningRate)];
>>>>>>> 9f53995efe149180065e31e589a4fd5232ca845c
%     opts.trackerName = 'LiteCRTv3';
    opts.videoAttr = 'tc128';
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [];
    opts.settingFcn = @(x) setting_lcrt_forfun(x, learningRate);
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end