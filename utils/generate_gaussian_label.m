function labels = generate_gaussian_label(sz, outputSigma, targetSize)
    
    [rs, cs] = ndgrid((1:sz(2)) - ceil(sz(2)/2), (1:sz(1)) - ceil(sz(1)/2));

    labels = exp(-0.5 * (((rs.^2/outputSigma(2)^2 + cs.^2/outputSigma(1)^2) ))); 
    
    labels = single(labels);
    labels = (labels - min(labels(:))) / (max(labels(:)) - min(labels(:)) + eps); 
    
%     labels = single(labels);
end

