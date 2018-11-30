function img = map_to_jpg(imgmap, range, colorMap)
% copy from Bolei Zhou
imgmap = double(imgmap);
if(~exist('range', 'var') || isempty(range)), range = [min(imgmap(:)) max(imgmap(:))]; end

heatmap_gray = mat2gray(imgmap, range);
heatmap_x = gray2ind(heatmap_gray, 256);
heatmap_x(isnan(imgmap)) = 0;

if(~exist('colorMap', 'var'))
    img = ind2rgb(heatmap_x, jet(256));
else
    img = ind2rgb(heatmap_x, eval([colorMap '(256)']));
end