clc; clear all; close all;
warning off;

opts.videoPath = 'D:/Dataset/Video/UAV123/';
opts.savePath = 'F:\Research\tracker_zoo\Evaluation\results\My-Work\LiteCRT/';
opts.netPath = 'backnet/vggm-conv1.mat';

opts.trackerName = 'uav123_LiteCRT-test';
opts.videoAttr = 'UAV123';
opts.verbose = false;
opts.useGpu = true;
opts.saveResult = true;
opts.videoList = [];
opts.settingFcn = @setting_lcrt_forfun;
opts.trackerFcn = @tracker_OPE;

[~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

eval_tracker(opts);