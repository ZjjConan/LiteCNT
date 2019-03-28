clc; clear all; close all;
warning off;

opts.videoPath = 'D:/Dataset/Video/OTB2015/';
opts.savePath = 'F:\Research\tracker_zoo\Evaluation\results/My-Work/LiteCRT/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('otb_distribute.mat');
for learningRate = [1e-5]
    opts.trackerName = ['otb2015_BaselineCRT_' num2str(learningRate)];
    opts.videoAttr = 'UAV123';
    opts.verbose = true;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [];
    opts.settingFcn = @(x) setting_lcnt_uav123(x, learningRate);
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end