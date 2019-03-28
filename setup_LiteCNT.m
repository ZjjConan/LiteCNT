function setup_LiteCNT()
    clc; clear all; close all
    if ispc
        lib_path = 'D:/Libraries/';
    elseif isunix
        lib_path = '/media/zjjconan/Experiments/Libraries/'; 
    end

    matconvnet_path = fullfile(lib_path, 'matconvnet');
    run([matconvnet_path '/matlab/vl_setupnn']);
        
    root = fileparts(fileparts(mfilename('fullpath'))) ;
    addpath(fullfile(root, 'LiteCNT')) ;
    
    addpath(genpath([root '/LiteCNT/tracking/']));
    addpath(fullfile(root, '/LiteCNT/utils/'));
    addpath(fullfile(root, '/LiteCNT/layers/'));
    addpath(fullfile(root, '/LiteCNT/headnet/'));

    % add lx tracker utils
    if ispc
        addpath(genpath('F:/Research/tracker_zoo/lx-trackTools'));
    elseif isunix
        addpath(genpath('/media/zjjconan/Experiments/tracker_zoo/lx-trackTools'));
    end
    
    addpath('D:/Libraries/PDollar-toolbox/channels/');
    
end