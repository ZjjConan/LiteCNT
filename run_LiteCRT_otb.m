clc; clear all; close all;
warning off;

opts.videoPath = 'D:/Dataset/Video/OTB';
opts.savePath = 'F:\Research\tracker_zoo\Evaluation\results\My-Work\LiteCRT/';
opts.netPath = 'backnet/vggm-conv1.mat';

opts.trackerName = 'LiteCRT-test';
opts.videoAttr = 'OTB2015';
opts.verbose = false;
opts.useGpu = true;
opts.saveResult = true;
opts.videoList = [];
opts.settingFcn = @setting_lcrt_forfun;
opts.trackerFcn = @tracker_lcrt;

[~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

eval_tracker(opts);