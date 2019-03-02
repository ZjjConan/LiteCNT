clc; clear all; close all;

opts.videoPath = 'D:\Dataset\Video\OTB\';
opts.videoName = 'OTB2015';
opts.trkResult = 'F:\Research\tracker_zoo\Evaluation\results\OTB/';
opts.savePics = 'Picture/';


index = [33];

opts.tracker1 = 'CFNet-Conv1-GT';
opts.tracker2 = 'CFNet-Conv1-GT-TN';

[videos, cfgReader] = loadVideos(opts.videoName);

% 26 72 79

load projMatrix_sa.mat

for i = index
    % load groundtruth
    cfg = cfgReader(opts.videoPath, videos{i}); 

    load([opts.trkResult '/' opts.tracker1 '/' videos{i} '_' opts.tracker1 '.mat']);  
    tracker1_featr = feature;
    tracker1_templ = template;
    tracker1_resps = response;
    tracker1_crops = searchArea;
    
    load([opts.trkResult '/' opts.tracker2 '/' videos{i} '_' opts.tracker2 '.mat']);  

    tracker2_featr = feature;
    tracker2_templ = template;
    tracker2_resps = response;
    tracker2_crops = searchArea;

    info = [];
    saveDir = [opts.savePics videos{i} '/'];
    mkdir(saveDir);
    
    for j = [50:100]
        featr1_color = showColorLayer(tracker1_featr{j}, projMatrix.meanFeat', projMatrix.V);
        featr2_color = showColorLayer(tracker2_featr{j}, projMatrix.meanFeat', projMatrix.V);
        
        featr1_color = imresize(featr1_color, [255, 255]);
        featr2_color = imresize(featr2_color, [255, 255]);
        
        resps1_color = map2jpg(tracker1_resps{j}, [], 'jet');
        resps2_color = map2jpg(tracker2_resps{j}, [], 'jet');
        
        resps1_color = imresize(resps1_color, [255, 255]);
        resps2_color = imresize(resps2_color, [255, 255]);
        
        imshow(tracker2_crops{j}/255);
        
        nframe = sprintf('%04d', j);
        text(10, 18, nframe, 'Color', 'y', 'FontWeight', 'bold', 'FontSize', 30);
        drawnow;
        pause(0.0001);
        
        f = getframe();
        imwrite(f.cdata,  fullfile(saveDir, [nframe '_tracker2_crops.png']));
        
        imshow(1 - single(featr1_color)/255);
        f = getframe();
        imwrite(f.cdata,  fullfile(saveDir, [nframe '_featr1_color.png']));
        
        imshow(1 - single(featr2_color)/255);
        f = getframe();
        imwrite(f.cdata,  fullfile(saveDir, [nframe '_featr2_color.png']));
        
        imshow(resps1_color);
        f = getframe();
        imwrite(f.cdata,  fullfile(saveDir, [nframe '_resps1_color.png']));
        
        imshow(resps2_color);
        f = getframe();
        imwrite(f.cdata,  fullfile(saveDir, [nframe '_resps2_color.png']));
        
        
%         templ1_color = showColorLayer(tracker1_templ{j}, projMatrix.meanFeat', projMatrix.V);
%         templ2_color = showColorLayer(tracker2_templ{j}, projMatrix.meanFeat', projMatrix.V);
%         
%         imshow(templ1_color);
%         f = getframe();
%         imwrite(f.cdata,  fullfile(saveDir, [nframe '_templ1_color.png']));
%         
%         imshow(templ2_color);
%         f = getframe();
%         imwrite(f.cdata,  fullfile(saveDir, [nframe '_templ2_color.png']));
        
%         figure(1); imshow([tracker1_crops{j}/255 1 - single(featr1_color)/255 resps1_color]);
%         figure(2); imshow([tracker2_crops{j}/255 1 - single(featr2_color)/255 resps2_color]);
        
       
    end
    
end