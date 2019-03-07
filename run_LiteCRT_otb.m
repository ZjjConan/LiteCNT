clc; clear all; close all;
warning off;

opts.videoPath = '/media/zjjconan/Data/Dataset/Video/Temple-color-128/';
opts.savePath = '/media/zjjconan/Experiments/tracker_zoo/Evaluation/results/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('video_distribute.mat');

for learningRate = [7e-6]
%     learningRate = 8e-6;
    opts.trackerName = ['tpl_LiteCRTv3-TSE2-' num2str(learningRate)];
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