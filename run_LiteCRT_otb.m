clc; clear all; close all;
warning off;

opts.videoPath = 'D:/Dataset/Video/LaSOT/';
opts.savePath = 'F:/Research/tracker_zoo/Evaluation/results/My-Work/LiteCRT/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('video_distribute.mat');

for learningRate = [1e-5 9e-6 8e-6 7e-6 6e-6 5e-6 4e-6 3e-6]
    opts.trackerName = ['lasot_LiteCRTv3-TSE-' num2str(learningRate)];
%     opts.trackerName = 'LiteCRTv2';
%     opts.trackerName = 'LiteCRTv3';
    opts.videoAttr = 'LaSOT-Test';
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