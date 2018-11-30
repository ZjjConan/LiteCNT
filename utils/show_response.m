function show_response(patch, score, alpha, state)
    patch = gather(bsxfun(@plus, patch(:,:,:,2), state.opts.averageImage));
    patch = patch / 255;
    score = map_to_jpg(score(:,:,2), [], 'jet');
    score = imresize(score, [size(patch,1) size(patch,2)]);
%     imshow((1 - alpha) * patch + alpha * score); 
    imwrite(patch, sprintf('data/cen/%04d_org.png', state.currFrame));
    imwrite(score, sprintf('data/cen/%04d_res.png', state.currFrame));
%     patch = uint8(((1 - alpha) * patch + alpha * score)*255);
%     imwrite(patch, sprintf('data/org/%04d.png', state.currFrame));
end

