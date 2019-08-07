function cfg = TC(base_path, video)


	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\'
		base_path(end+1) = '/';
    end
    
	video_path = [base_path video '/'];
	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path video '_gt.txt'];
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

    filename = [video_path video '_att.txt'];
	f = fopen(filename);
    cfg.att = textscan(f, '%s');
    cfg.att = cfg.att{1};
    fclose(f);
	
    filename = [video_path video '_frames.txt'];
	f = fopen(filename);
    frames = textscan(f, '%f,%f');
    cfg.startFrame = frames{1};
    cfg.endFrame = frames{2};
    cfg.annoBegin = frames{1};
    cfg.annoEnd = frames{2};
    fclose(f);
    
	%from now on, work in the subfolder where all the images are
	video_path = [video_path 'img/'];

	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.


     
	idx = find(strcmpi(video, frames(:,1)));
	
	if isempty(idx),
		%general case, just list all images
		img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});
	else
		%list specified frames. try png first, then jpg.
		if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
		elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
    end
    %eliminate frame 0 if it exists, since frames should only start at 1
	img_files(strcmp('00000000.jpg', img_files)) = [];
    if ~strcmpi(img_files{1}, [sprintf('%04d', cfg.startFrame) '.jpg'])
        img_files = img_files(cfg.startFrame:cfg.endFrame);
%         ground_truth = ground_truth(cfg.startFrame:cfg.endFrame, :);
    end
    cfg.img_files = strcat(video_path, img_files);
    cfg.ground_truth = ground_truth;

end

