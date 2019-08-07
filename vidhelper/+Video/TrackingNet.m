function cfg = TrackingNet(base_path, video)
	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\'
		base_path(end+1) = '/';
    end
    
	video_path = [base_path 'frames/' video '/'];
	%try to load ground truth from text file (Benchmark's format)
	filename = [base_path 'anno/' video '.txt'];
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

    cfg.startFrame = 1;
    cfg.endFrame = 'trackingnet-notused';
    cfg.annoBegin = 1;
    cfg.annoEnd = size(ground_truth, 1);
		
    img_files = dir([video_path '*.png']);
	if isempty(img_files),
		img_files = dir([video_path '*.jpg']);
		assert(~isempty(img_files), 'No image files to load.')
	end
	img_files = sort({img_files.name});
    
    [~, idx] = sort(str2double(regexp(img_files, '\d+', 'match', 'once' )));
    img_files = img_files(idx) ;
    
    cfg.img_files = strcat(video_path, img_files);
    cfg.ground_truth = ground_truth;

end

