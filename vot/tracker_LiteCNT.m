% tracker config
tracker_label = 'LiteCNT';
tracker_interpreter = 'matlab';
tracker_trax = true;

% change you tracker path
tracker_path = '###############################';

% the model path
netPath = fullfile(tracker_path, 'models/vggm-conv1.mat');

% setting function, numRegions = 2
settingFcn = '@(x) setting_lcnt_default(x, 2)';

% tracker command
fullCommand = ['run_LiteCNT_vot(', '''' netPath, ''',', settingFcn, ')'];

tracker_command = generate_matlab_command(fullCommand, {fullfile(tracker_path, 'vot')});

% link your cudnn library, ie, cudnn/lib/x64
tracker_linkpath = {'###############################'};