function feature = normalize_feature(feature, opts)
    [r, c, d, n] = size(feature);
    scaleFactor = sqrt(r*c*d/sum(reshape(feature, [], 1, 1, n).^2, 1) + eps);
    feature = feature .* scaleFactor;
end

