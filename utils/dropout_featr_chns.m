function featr = dropout_featr_chns(featr)
    chns = size(featr, 3);
    idx = randperm(chns, round(0.2*chns));
    featr(:,:,idx) = 0;
    featr = featr / (1 - 0.2);
end

