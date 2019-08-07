function cfg = LaSOT(base_path, video)
	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
    end
    category_dir = strsplit(video, '-');
    category_dir = category_dir{1};
	video_path = [base_path '/data/' category_dir '/' video '/'];
	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path 'groundtruth.txt'];
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
    
    
    f = fopen([video_path 'full_occlusion.txt']);
    cfg.occlusion = textscan(f, '%f,');
    cfg.occlusion = cfg.occlusion{1};
    fclose(f);
    
    f = fopen([video_path 'out_of_view.txt']);
    cfg.out_of_view = textscan(f, '%f,');
    cfg.out_of_view = cfg.out_of_view{1};
    fclose(f);
    
		
%     cfg.att = load([video_path 'cfg.mat']);
%     cfg.att = cfg.att.seq;
	
	%from now on, work in the subfolder where all the images are
	video_path = [video_path 'img/'];

    cfg.startFrame = 1;
    cfg.annoBegin = 1;       
	
    img_files = dir([video_path '*.png']);
	if isempty(img_files),
		img_files = dir([video_path '*.jpg']);
		assert(~isempty(img_files), 'No image files to load.')
	end
	img_files = sort({img_files.name});
	img_files = cellstr(img_files);
    %eliminate frame 0 if it exists, since frames should only start at 1
	img_files(strcmp('00000000.jpg', img_files)) = [];
    img_files = strcat(video_path, img_files);

    cfg.img_files = img_files;
    cfg.ground_truth = ground_truth;
    
    
    fclose all;
end

