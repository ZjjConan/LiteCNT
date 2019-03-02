clc; clear all; close all;
warning off;

opts.videoPath = '/media/zjjconan/Data/Dataset/Video/Temple-color-128/';
opts.savePath = '/media/zjjconan/Experiments/tracker_zoo/Evaluation/results/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('video_distribute.mat');

<<<<<<< HEAD
for learningRate = [5e-6]
    opts.trackerName = ['tpl_LiteCRTv2'];
    opts.videoAttr = 'tc128';
=======
for learningRate = [1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5]
    opts.trackerName = ['lasot_LiteCRTv2-BBR-' num2str(learningRate)];
    opts.videoAttr = 'LaSOT-Test';
>>>>>>> refs/remotes/origin/master
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = harder;
    opts.settingFcn = @(x) setting_lcrt_forfun(x, learningRate);
%      opts.settingFcn = @setting_lcrt_cvpr2019;
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end