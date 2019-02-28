function featr = base_extract_feature(net, img, opts)
    net.mode = 'test';
    
    if strcmpi(net.device, 'gpu')
        img = gpuArray(img);
    end
    
    net.eval({'input', img});
    featr = net.vars(end).value;
    
    if ~isempty(opts.cosineWindow)
        featr = bsxfun(@times, featr, opts.cosineWindow);
    end
    
    if opts.normalize
        featr = normalize_feature(featr);
    end

end

