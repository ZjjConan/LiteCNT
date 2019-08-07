function seqNames = LaSOT(datapath)
    subdirs = get_dir_names(datapath);
    categories = get_dir_names(fullfile(datapath, subdirs{2}));
    assert(numel(categories) == 70);
    
    subnames = textscan(fopen((fullfile(datapath, 'test_set.txt')), 'r'), '%s'); 
    filenames{1} = subnames{1};
    filenames{1} = sort(filenames{1});
    set{1} = 2*ones(numel(filenames{1}), 1);

    subnames = textscan(fopen((fullfile(datapath, 'train_set.txt')), 'r'), '%s'); 
    filenames{2} = subnames{1};
    filenames{2} = sort(filenames{2});
    set{2} = ones(numel(filenames{2}), 1);
    
    seqNames.videos = cat(1, filenames{:});
    seqNames.set = cat(1, set{:});
end

