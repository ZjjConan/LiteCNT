function eval_tracker(varargin)
    opts.netPath = [];
    opts.savePath = [];
    opts.trackerName = [];
    opts.videoPath = 'D:\Dataset\Video\OTB\';
    opts.videoAttr = 'OTB2015';
    opts.verbose = false;
    opts.useGpu = 1;
    opts.runFileName = '';
    opts.settingFcn = [];
    opts.saveResult = true;
    opts.videoList = [];
    opts.trackerFcn = [];
    opts = vl_argparse(opts, varargin);
    
    % create dir if not exist
    if ~exist([opts.savePath opts.trackerName], 'file')
        mkdir([opts.savePath opts.trackerName]);
    end

    % get all videos' name and loaderFunc
    [videos, cfgReader] = config_dataset(opts.videoAttr, opts.videoPath);
    % get opts
    trkOpts = opts.settingFcn(opts);
    
    if isempty(opts.videoList)
        opts.videoList = 1:numel(videos);
    end
    
    if opts.useGpu
        reset(gpuDevice); 
    end
 
    for i = opts.videoList        
        if strcmpi(videos{i}, 'human4'), videos{i} = 'Human4-2'; end
%         reset(gpuDevice);
%         if exist([opts.savePath opts.trackerName '/' videos{i} '_' opts.trackerName '.mat'], 'file')
%             continue;
%         end
      
        cfg = cfgReader(opts.videoPath, videos{i});
        fprintf('%s: process %d -- %s seq ...', opts.runFileName, i, videos{i});
        [result, fps, trkMemory, nreset] = opts.trackerFcn(cfg, trkOpts);
        fprintf(' pmem %s fps info -- all [%.2f] / det [%.2f] / dup [%.2f]\n', pmem(trkMemory.modelSize), fps.all, fps.det, fps.dup);
        results{1}.type = 'rect';
        results{1}.len = size(result, 1);
        results{1}.annoBegin = cfg.annoBegin;
        results{1}.startFrame = cfg.startFrame;
        results{1}.res = result;
        results{1}.fps = fps.all;
        results{1}.reset = nreset;
        results{1}.modelSize = trkMemory.modelSize;
        
        if opts.saveResult
            save([opts.savePath opts.trackerName '/' videos{i} '_' opts.trackerName '.mat'], 'results');
        end
    end
end