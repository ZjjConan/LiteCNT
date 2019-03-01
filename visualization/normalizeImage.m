function [nimg, info] = normalizeImage(img, inputSize)

    [rows, cols, ~] = size(img);
    info.rscale = inputSize(1) / rows;
    info.cscale = inputSize(2) / cols;
    nimg = imresize(gather(img), inputSize);
    
%     nboxes = boxes;
%     nboxes(:, [1,3]) = nboxes(:, [1,3]) * cscale;
%     nboxes(:, [2,4]) = nboxes(:, [2,4]) * rscale;

end

