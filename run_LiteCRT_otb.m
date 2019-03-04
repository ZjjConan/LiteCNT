clc; clear all; close all;
warning off;

opts.videoPath = '/media/zjjconan/Data/Dataset/Video/LaSOT/';
opts.savePath = '/media/zjjconan/Experiments/tracker_zoo/Evaluation/LaSOT/';
opts.netPath = 'backnet/vggm-conv1.mat';

load('video_distribute.mat');

for learningRate = [3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5]
    opts.trackerName = ['lasot_LiteCRTv2-' num2str(learningRate)];
<<<<<<< HEAD
=======
%     opts.trackerName = 'LiteCRTv2';
>>>>>>> 807e48cb1da7bf1ae776981d99f25d64fd2867c4
    opts.videoAttr = 'LaSOT-Test';
    opts.verbose = false;
    opts.useGpu = true;
    opts.saveResult = true;
    opts.videoList = [harder];
    opts.settingFcn = @(x) setting_lcrt_forfun(x, learningRate);
%      opts.settingFcn = @setting_lcrt_cvpr2019;
    opts.trackerFcn = @tracker_OPE;

    [~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));

    eval_tracker(opts);
end