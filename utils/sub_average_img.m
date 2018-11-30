function img = sub_average_img(img, averageImage)
    if ~isempty(averageImage)
        if isscalar(averageImage)
            img = img - averageImage;
        else
            img = bsxfun(@minus, img, averageImage);
        end
    end
end

