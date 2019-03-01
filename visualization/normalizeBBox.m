function nboxes = normalizeBBox(boxes, info)
    nboxes = boxes;
    nboxes(:, [1,3]) = nboxes(:, [1,3]) * info.cscale;
    nboxes(:, [2,4]) = nboxes(:, [2,4]) * info.rscale;
end

