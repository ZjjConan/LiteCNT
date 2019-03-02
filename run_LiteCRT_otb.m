clc; clear all; close all;
warning off;

opts.videoPath = '/media/zjjconan/Data/Dataset/Video/Temple-color-128/';
opts.savePath = '/media/zjjconan/Experiments/tracker_zoo/Evaluation/results/';
opts.netPath = 'backnet/vggm-conv1.mat';


for learningRate = [5e-6]
    opts.trackerName = ['tpl_LiteCRTv2'];
    opts.videoAttr = 'tc128';
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [];
    opts.settingFcn = @(x) setting_lcrt_forfun(x, learningRate);
%      opts.settingFcn = @setting_lcrt_cvpr2019;
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end