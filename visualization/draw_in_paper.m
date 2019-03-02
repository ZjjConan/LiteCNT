clear all; close all;
% startup;
addpath(genpath('F:/Research/tracker_zoo/lx-trackTools/'));
clc

opts.savePath = 'F:/Research/tracker_zoo/Evaluation/results/OTB/';
opts.trackerName = 'CFNet-Conv1-GT';
opts.videoPath = 'D:/Dataset/Video/OTB/';
opts.videoAttr = 'OTB2015'; 
opts.verbose = false;

opts.trackerFcn = @tracker_gt;

% get all videos' name and loaderFunc
[videos, cfgReader] = loadVideos(opts.videoAttr, opts.videoPath);

tracker_par.join.method = 'corrfilt';
tracker_par.net = 'cfnet-conv1_e75.mat';
tracker_par.net_gray = 'cfnet-conv1_gray_e55.mat';
tracker_par.scaleStep = 1.0355;
tracker_par.scalePenalty = 0.9825;
tracker_par.scaleLR = 0.7;
tracker_par.wInfluence = 0.2375;
tracker_par.zLR = 0.0058;

index = [];

   
% create dir if not exist
if ~exist([opts.savePath opts.trackerName], 'file')
    mkdir([opts.savePath opts.trackerName]);
end

tracker_par.TNLearningRate = 0.06;

tracker_par.visualization = opts.verbose;
tracker_par.gpus = [1];
tracker_par.paths = env_paths_tracking();
tracker_par.track_lost = [];
tracker_par.subMean = false;

% 33: David;
index = [72];

for i = index
    if strcmpi(videos{i}, 'human4') 
        videos{i} = 'Human4-2';
    end

    gpuDevice(1);
    pause(0.1);
    cfg = cfgReader(opts.videoPath, videos{i});

    region = cfg.ground_truth(1, :);
    [cx, cy, w, h] = get_axis_aligned_BB(region);
    tracker_par.targetPosition = [cy cx]; % centre of the bounding box
    tracker_par.targetSize = [h w];
    tracker_par.imgFiles = vl_imreadjpeg(cfg.img_files, 'NumThreads', 4);
    tracker_par.ground_truth = cfg.ground_truth;
    tracker_par.startFrame = 1;

    fprintf('process %d -- %s seq ...', i, videos{i});
    [result, fps, searchArea, template, feature, response] = opts.trackerFcn(tracker_par);
    fprintf('done, fps: %.2f\n', fps);
    results{1}.type = 'rect'; 
    results{1}.len = size(result, 1);
    results{1}.annoBegin = cfg.annoBegin;
    results{1}.startFrame = cfg.startFrame;
    results{1}.res = result;
    results{1}.fps = fps;
    save([opts.savePath opts.trackerName '/' videos{i} '_' opts.trackerName '.mat'], 'results', ...
        'searchArea', 'template', 'feature', 'response');
end
    

