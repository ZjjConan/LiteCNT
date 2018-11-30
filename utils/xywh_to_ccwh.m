function [p, s] = xywh_to_ccwh(region)
    p = region(:, 1:2) + region(:, 3:4)/2;
    s = region(:, 3:4);
end

