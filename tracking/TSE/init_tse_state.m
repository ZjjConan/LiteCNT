function [tsEstimator, sparams] = init_tse_state(targetSize, sparams)
    % contraint the target size for efficient tracking
    if max(targetSize) > sparams.TSEMaxTargetSize
        resizedRatio = max(targetSize) ./ sparams.TSEMaxTargetSize;
%     elseif min(targetSize) < aparams.TSEMinTargetSize
%         resizedRatio = min(targetSize) ./ aparams.TSEMaxTargetSize;
    else
        resizedRatio = 1;
    end
    
    scaledTargetSize = max(round(targetSize ./ resizedRatio), [8 8]);
    
    numScales = sparams.TSENumScaleFilters;
    scaleStep = sparams.TSEScaleFactor;

    sigma = sparams.TSENumInterpScales * sparams.TSESigmaFactor;
    scaleExp = (-floor((numScales-1)/2):ceil((numScales-1)/2)) * sparams.TSENumInterpScales/numScales;
    scaleShift = circshift(scaleExp, [0 -floor((numScales-1)/2)]);

    interpScaleExp = -floor((sparams.TSENumInterpScales-1)/2):ceil((sparams.TSENumInterpScales-1)/2);
    interpScaleShift = circshift(interpScaleExp, [0 -floor((sparams.TSENumInterpScales-1)/2)]);

    tsEstimator.scaleSizeFactors = scaleStep .^ scaleExp;
    tsEstimator.interpScaleFactors = scaleStep .^ interpScaleShift;

    ys = exp(-0.5 * (scaleShift.^2) /sigma^2);
    tsEstimator.yf = single(fft(ys));
    tsEstimator.window = single(hann(size(ys,2)))';
    
    
    gridGenerator = dagnn.AffineGridGenerator('Ho', scaledTargetSize(2), 'Wo', scaledTargetSize(1));
    sparams.gridGenerator = gridGenerator;
    sparams.numScales = numScales;
end

