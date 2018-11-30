function nbbox = project_bbox(bbox, cropCoord, cropRatio, imageSize)
%     samples(:, 3:4) = samples(:, 1:2) + samples(:, 3:4);
%     samples = bbox_clip(samples, opts.imageSize);
% %     ok = is_valid_bbox(samples, cropCoord);
%     samples = bbox_project(samples, cropCoord, cropRatio);

    nbbox = bbox;
    nbbox(:, 1:2) = bsxfun(@minus, bbox(:, 1:2), cropCoord(1:2));
    nbbox(:, [1,3]) = nbbox(:, [1,3]) * cropRatio(1);
    nbbox(:, [2,4]) = nbbox(:, [2,4]) * cropRatio(2);
%     nbbox = clip_bbox(nbbox, imageSize);
end

