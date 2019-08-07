function dirNames = get_dir_names(datapath)
    dirNames = dir(datapath);
    dirNames = {dirNames.name}';
    idx = strcmpi(dirNames, '.') | strcmpi(dirNames, '..');
    dirNames(idx) = [];
end

