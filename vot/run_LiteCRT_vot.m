function run_LiteCRT_vot(netPath, settingFcn)

    cleanup = onCleanup(@() exit() );

    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));
    
try    
    setup_LiteCRT_vot;
    
    traxserver('setup', 'polygon', {'path'});
    
    opts.netPath = netPath;
    opts.verbose = 0;
    opts.useGpu = 1;
    
    trackerOpts = settingFcn(opts);
    
    [image, region] = traxserver('wait');
    [cx, cy, w, h] = get_axis_aligned_BB(region);
    region = [cx-w/2, cy-h/2, w, h];
    
    img = read_img(image);
    
    if opts.useGpu
        img = gpuArray(single(img));
    else
        img = single(img);
    end
    
    state = trackerOpts.state_initialize(img, region, trackerOpts);

    % warm up
    state = trackerOpts.state_warmup(state);
        
    tracker_initialized = false;
    while true
        if ~tracker_initialized
            state = trackerOpts.initialize(state, img);
            tracker_initialized = true;
        else
            [image, region] = traxserver('wait');
            
            if isempty(image)
                break
            end
            
            img = read_img(image);     
            
            if  opts.useGpu
                img = gpuArray(single(img));
            else
                img = single(img);
            end
           
            state = trackerOpts.track(state, img);
            state = trackerOpts.update(state, img);
        end

        if isempty(state.result(state.currFrame, :))
            state.result(state.currFrame, :) = [0, 0, 1, 1];
        end
        
        parameters = struct();
        parameters.confidence = state.targetScore;
        
        traxserver('status', double(state.result(state.currFrame, :)), parameters);
    end
%     % release gpu
%     state = trackerOpts.release(state);
    
    traxserver('quit');
    
catch err
    [wrapper_pathstr, ~, ~] = fileparts(mfilename('fullpath'));
    cd_ind = strfind(wrapper_pathstr, filesep());
    VOT_path = wrapper_pathstr(1:cd_ind(end));
    
    error_report_path = [VOT_path '/error_reports/'];
    if ~exist(error_report_path, 'dir')
        mkdir(error_report_path);
    end
    
    report_file_name = [error_report_path 'LiteCRT' datestr(now,'_yymmdd_HHMM') '.mat'];
    
    save(report_file_name, 'err')
    
    rethrow(err);
end

