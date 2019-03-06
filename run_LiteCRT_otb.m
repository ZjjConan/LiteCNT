clc; clear all; close all;
warning off;

opts.videoPath = '/media/zjjconan/Data/Dataset/Video/Temple-color-128/';
opts.savePath = '/media/zjjconan/Experiments/tracker_zoo/Evaluation/results/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('video_distribute.mat');

<<<<<<< HEAD
for learningRate = [8e-6]
%     learningRate = 8e-6;
    opts.trackerName = ['tpl_LiteCRTv3-TSE2-' num2str(learningRate)];
%     opts.trackerName = 'LiteCRTv3';
    opts.videoAttr = 'tc128';
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [];
=======
for learningRate = [1e-5 9e-6 8e-6 7e-6 6e-6 5e-6 4e-6 3e-6]
    opts.trackerName = ['lasot_LiteCRTv3-TSE-' num2str(learningRate)];
%     opts.trackerName = 'LiteCRTv2';
%     opts.trackerName = 'LiteCRTv3';
    opts.videoAttr = 'LaSOT-Test';
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = harder;
>>>>>>> f9f9cd232a2e7e4a8ae1ff358930a8b879abc026
    opts.settingFcn = @(x) setting_lcrt_forfun(x, learningRate);
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end