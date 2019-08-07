function cfg = UAV123(base_path, video)
    configs = uav123_config();
    for i = 1:numel(configs)
        if strcmpi(configs{i}.name, video)
            break
        end
    end
    
    cfg.startFrame = configs{i}.startFrame;
    cfg.endFrame = configs{i}.endFrame;
    cfg.annoBegin = cfg.startFrame;
    cfg.annoEnd = cfg.endFrame;
    
    vname = strsplit(video, '_');
    if strcmpi(vname{end}, 's')
        vname = {video};   
    end
    video_path = fullfile(base_path, 'data_seq', vname{1});

	filename = [base_path 'anno/UAV123/' video '.txt'];
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]
	try
		ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  %#ok, try different format (no commas)
		frewind(f);
		ground_truth = textscan(f, '%f %f %f %f');  
	end
	ground_truth = cat(2, ground_truth{:});
	fclose(f);
		
    cfg.att = load([base_path 'anno/UAV123/att/' video '.txt']);
	
	img_files = dir([video_path '/*.png']);
    if isempty(img_files),
        img_files = dir([video_path '/*.jpg']);
        assert(~isempty(img_files), 'No image files to load.')
	end
	img_files = sort({img_files.name});
	
    %eliminate frame 0 if it exists, since frames should only start at 1
	img_files(strcmp('00000000.jpg', img_files)) = [];
    img_files = fullfile(video_path, img_files);

    cfg.img_files = img_files(cfg.startFrame:cfg.endFrame)';
    cfg.ground_truth = ground_truth;

end

