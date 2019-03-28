function [patch, label] = data_augmenter(img, label, state)

    [targetPos, targetSize] = xywh_to_ccwh(state.targetRect); 
    patch = cell(length(state.aparams)+1, 1);
    label = repmat(label, 1, 1, 1, length(state.aparams)+1);
    patch{1} = crop_roi(img, state.targetRect, state.gparams);
    scaledRatio = state.scaledRatio;
%     scaleRatio = [scaleRatio scaleRatio];
    
    for i = 1:length(state.aparams)
        pos = targetPos;
        sz = targetSize;
        gparams = state.gparams;
        % copy from UPDT
        switch state.aparams(i).type
            case 'fliplr'
                img_ = fliplr(img);
                pos(1) = size(img_, 2) - pos(1) + 1;
            case 'dropout'
                img_ = img;
            case 'shift'
                img_ = img;
                shift_params = state.aparams(i).param ./ scaledRatio;
                pos = pos + shift_params;
                shift_params = - shift_params ./ state.gparams.subStride;
                label(:,:,:,i+1) = circshift(label(:,:,:,i+1), round(shift_params([2,1])));
            case 'blur'
                img_ = imgaussfilt(img, state.aparams(i).param);
                    
            case 'rot'
                    
                theta = state.aparams(i).param;

                T_trans1 = [1, 0, 0; ...
                            0, 1, 0; ...
                            -pos(1), -pos(2), 1];

                T_rot = [cosd(theta), sind(theta), 0; ...
                        -sind(theta), cosd(theta),  0; ...
                         0, 0, 1];

                T_trans2 = [1, 0, 0; ...
                            0, 1, 0; ...
                            pos(1), pos(2), 1];

                tform = affine2d(T_trans1*T_rot*T_trans2);
                [img_, ref] = imwarp(gather(img), tform) ;

                        % Find transformed location
                [x1, y1] = transformPointsForward(tform,pos(1),pos(2));

                pos(1) = round(x1 - ref.XWorldLimits(1));
                pos(2) = round(y1 - ref.YWorldLimits(1));
                gparams.imageSize = [size(img_, 2) size(img_, 1)];
            otherwise
                error('error type of augmentation');
        end
        bbox = ccwh_to_xywh(pos, sz);
        patch{i+1} = crop_roi(img_, bbox, gparams);
    end
    patch = cat(4, patch{:});
end

