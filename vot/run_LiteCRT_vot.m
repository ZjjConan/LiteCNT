function run_LiteCRT_vot(initModel, settingFcn)

    cleanup = onCleanup(@() exit() );

    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));
    
try    
    setup_liteCRT_vot;
    
    opts.initModel = initModel;
    opts.verbose = 0;
    opts.useGpu = 1;
    
    runOpts = settingFcn(opts);
     
    % warm up
    for i = 1:10
        state = runOpts.state_initialize(randn(300, 300, 3, 'single'), single([50 50 120 120]), runOpts);
    end
    
%     [wrapper_pathstr, ~, ~] = fileparts(mfilename('fullpath'));
%     cd_ind = strfind(wrapper_pathstr, filesep());
%     VOT_path = wrapper_pathstr(1:cd_ind(end));
%     save([VOT_path '/runOpts.mat'], 'runOpts');

    traxserver('setup', 'polygon', {'path'});
    while true
        
        [image, region] = traxserver('wait');

        if isempty(image)
            break
        end

        img = read_img(image);

        if ~isempty(region)
            [cx, cy, w, h] = get_axis_aligned_BB(region);
            region = [cx-w/2, cy-h/2, w, h];
            state = runOpts.state_initialize(img, region, runOpts);
            state = runOpts.initialize(state, img);
        else
            state = runOpts.track(state, img);
            state = runOpts.update(state);
        end

        if isempty(state.result(state.currFrame, :))
            state.result(state.currFrame, :) = [0, 0, 1, 1];
        end
        
        parameters = struct();
        parameters.confidence = state.targetScore;
        
        traxserver('status', double(state.result(state.currFrame, :)), parameters);
    end

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

