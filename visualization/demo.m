

img=imread('D:\Dataset\Video\OTB\MotorRolling\img/0019.jpg');

imgS=single(img);
[imgH, imgW, nCh] = size(img);
if nCh==1
    img=repmat(img,[1, 1, 3]);
end


opts.netPath = '../backnet/vggm-conv1.mat';
opts.isDagNN = true;
opts.usePad = false;
opts.downsamplingFactor = 2;
opts.downsamplingMethod = 'avg';
opts.avgImage = single(reshape([122.6769, 116.67, 104.01], 1, 1, 3));

net = init_featrnet(opts);

% imgC = imresize(imgS, [255 255]); 

imgC = bsxfun(@minus, imgS, opts.avgImage);

z_out_id = net.getOutputs;
get_vars = @(net, ids) cellfun(@(id) net.getVar(id).value, ids, 'UniformOutput', false);

net.eval({net.getInputs{1}, imgC});

featrs = get_vars(net, z_out_id);
featrs = featrs{1};

load projMatrix_vggmc1avg.mat

j=1; % visualizing conv3

featColor=showColorLayer(featrs, projMatrix(j).meanFeat', projMatrix(j).V);
featColor=imresize(featColor, [imgH, imgW]);
imshow([img 255 - featColor]);

% figure, imagesc(featColor)
% figure, imagesc(255-featColor);
