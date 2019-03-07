clc; clear all; close all;
warning off;

opts.videoPath = 'D:/Dataset/Video/OTB/';
opts.savePath = 'F:/Research/tracker_zoo/Evaluation/results/My-Work/LiteCRT/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('otb_distribute.mat');

for learningRate = [9e-6]
%     learningRate = 1e-5;
    opts.trackerName = ['LiteCRTv3_8x6'];
%     opts.trackerName = 'LiteCRTv2';
%     opts.trackerName = 'LiteCRTv3';
    opts.videoAttr = 'OTB2015';
    opts.verbose = true;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [2];
    opts.settingFcn = @(x) setting_lcrt_uav123(x, learningRate);
%      opts.settingFcn = @setting_lcrt_cvpr2019;
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end