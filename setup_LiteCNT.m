function setup_LiteCNT()
    clc; clear all; close all

    % your matconvnet here
    lib_path = 'D:/Libraries/';
    
    matconvnet_path = fullfile(lib_path, 'matconvnet');
    run([matconvnet_path '/matlab/vl_setupnn']);
        
    root = fileparts(fileparts(mfilename('fullpath'))) ;
    addpath(fullfile(root, 'LiteCNT')) ;
    
    addpath(genpath([root '/LiteCNT/tracking']));
    addpath(fullfile(root, '/LiteCNT/utils'));
    addpath(fullfile(root, '/LiteCNT/layers'));
    addpath(fullfile(root, '/LiteCNT/headnet'));
    addpath(fullfile(root, '/LiteCNT/vidhelper'));

    addpath(genpath('F:/Research/tracker_zoo/lx-tracking-toolkits'));
end