function featr = tse_extract_feature(net, img, opts)
    net.mode = 'test';

    if strcmpi(net.device, 'gpu')
        img = gpuArray(img);
    end
    
    net.eval({'input', img});
    featr = net.vars(end).value;
    featr = reshape(featr, [], size(featr, 4));
%     img = gather(img);
%     for s = 1:size(img, 4)
%         temp_hog = fhog(img(:,:,:,s), 4);
%     
%         if s == 1
%             dim_scale = size(temp_hog,1)*size(temp_hog,2)*31;
%             featr = zeros(dim_scale, size(img, 4), 'single');
%         end
%     
%         featr(:,s) = reshape(temp_hog(:,:,1:31), dim_scale, 1);
%     end


%     if opts.normalize
%         featr = normalize_feature(featr);
%     end

end

