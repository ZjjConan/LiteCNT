function [hardest, harder, easy] = select_hard_videos(videoDir, videoAttr, trackers, threshold)

    videoDir = 'D:/Dataset/Video/OTB/';
    videoAttr{1} = 'OTB2015';
    videoAttr{2} = videoDir;
    
%     tracker_path = 'F:\Research\tracker_zoo\Evaluation\results\My-Work\LiteCRT/';
%     n = 1;
%     for lr = [1e-6 3e-6 5e-6 7e-6 9e-6 1e-5]
%         trackers{n}.path = tracker_path;
%         trackers{n}.name = ['lasot_LiteCRT-test-' num2str(lr)];
%         n = n + 1;
%     end
    
    trackers{1}.path = 'F:/Research/tracker_zoo/Evaluation/results/OTB/';
    trackers{1}.name = 'ECO';
    
    trackers{2}.path = 'F:/Research/tracker_zoo/Evaluation/results/OTB/';
    trackers{2}.name = 'C-COT';
    
    trackers{3}.path = 'F:/Research/tracker_zoo/Evaluation/results/OTB/';
    trackers{3}.name = 'MDNet';
    
    trackers{4}.path = 'F:/Research/tracker_zoo/Evaluation/results/OTB/';
    trackers{4}.name = 'DSLT';
    
    [videos, cfgLoader] = loadVideos(videoAttr{1}, videoAttr{2});
    
    threshold = 0.2;
   
    hardest = [];
    harder = [];
    easy = [];
    for i = 1:numel(videos)
                
        if strcmpi(videos{i}, 'Human4')
            videos{i} = 'Human4-2';
        end
        
        cfg = cfgLoader(videoDir, videos{i}); 
        score = zeros(numel(trackers), 1);
        for j = 1:numel(trackers)
            load([trackers{j}.path '/' trackers{j}.name '/' videos{i} '_' trackers{j}.name '.mat']);
            s = Evaluator.calcSuccRate(results{1}.res, cfg.ground_truth, 0:0.05:1);
            score(j) = mean(s);
        end
        
        if all(score < threshold)
            hardest(end+1) = i;
        elseif sum(score > 0.2 & score < 0.6) >= 3
            harder(end+1) = i;
        elseif all(score > 0.6)
            easy(end+1) = i;
        end
        
        
        
        fprintf('check %d / %d video performance\n', i, numel(videos));
    end
end

