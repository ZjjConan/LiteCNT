clc; clear all; close all;
warning off;

% change 
opts.videoPath = 'D:/Dataset/Video/OTB/';
opts.savePath = 'F:\Research\tracker_zoo\Evaluation\results/My-Work/LiteCRT/';
opts.netPath = 'models/vggm-conv1.mat';

opts.trackerName = 'otb2015_LiteCNT_amrconv';
opts.videoAttr = 'OTB2015';
opts.verbose = false;
opts.useGpu = true;
opts.saveResult = true;
opts.videoList = [];
opts.settingFcn = @(x) setting_lcnt_default(x, 2);
opts.trackerFcn = @tracker_OPE;

[~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

eval_tracker(opts);