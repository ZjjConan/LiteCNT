clc; clear all; close all;
warning off;

opts.videoPath = 'D:/Dataset/Video/OTB/';
opts.savePath = 'F:\Research\tracker_zoo\Evaluation\results/My-Work/LiteCRT/';
opts.netPath = 'backnet/vggm-conv1.mat';

for learningRate = [1e-5]
    opts.trackerName = ['otb100_LiteCRTv3_dataaug_' num2str(learningRate)];
    opts.videoAttr = 'OTB2015';
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [];
    opts.settingFcn = @(x) setting_lcnt_forfun(x, learningRate, 2);
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end