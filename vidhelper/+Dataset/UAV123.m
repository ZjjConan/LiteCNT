function seqNames = UAV123()
    configs = uav123_config();
    seqNames = {};
    for i = 1:numel(configs)
        seqNames{end+1} = configs{i}.name;
    end
    seqNames = seqNames';
end




