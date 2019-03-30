clc; clear all; close all;
warning off;

opts.videoPath = 'D:/Dataset/Video/tc128/';
opts.savePath = 'F:\Research\tracker_zoo\Evaluation\results/My-Work/LiteCRT/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('otb_distribute.mat');
for learningRate = [1e-5]
    opts.trackerName = ['tc128_LiteCRTv3_dataaug_' num2str(learningRate)];
    opts.videoAttr = 'tc128';
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [];
    opts.settingFcn = @(x) setting_lcnt_forfun(x, learningRate, 3);
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end