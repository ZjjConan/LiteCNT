function [videos, cfgReader] = config_dataset(dataset, datapath)


%%    
    if strcmpi(dataset, 'OTB2013')
        videos = Dataset.OTB2013();
        cfgReader = @Video.OTB;
    %
    elseif strcmpi(dataset, 'OTB2015')
        videos = Dataset.OTB2015();
        cfgReader = @Video.OTB;
    %
    elseif strcmpi(dataset, 'OTB50')
        videos = Dataset.OTB50();
        cfgReader = @Video.OTB;
    %
    elseif strcmpi(dataset, 'TC128')
        assert(nargin == 2, 'needs 2 input');
        videos = Dataset.TC128(datapath);
        cfgReader = @Video.TC;
    %
    elseif strcmpi(dataset, 'NFS30')
        assert(nargin == 2, 'needs 2 input');
        videos = Dataset.NFS(datapath);
        cfgReader = @(x,y) Video.NFS(x, y, 30);
    %
    elseif strcmpi(dataset, 'NFS240')
        assert(nargin == 2, 'needs 2 input');
        videos = Dataset.NFS(datapath);
        cfgReader = @(x,y) Video.NFS(x, y, 240);
    %
    elseif strcmpi(dataset, 'LaSOT-test') || strcmpi(dataset, 'LaSOT')
        assert(nargin == 2, 'needs 2 input');
        videos = Dataset.LaSOT(datapath);
        cfgReader = @Video.LaSOT;
        if strcmpi(dataset, 'LaSOT-test')
            videos = videos.videos(videos.set == 2);
        else
            videos = videos.videos;
        end
    %
    elseif strcmpi(dataset, 'TrackingNet')
        assert(nargin == 2, 'needs 2 input');
        videos = Dataset.TrackingNet(datapath);
        cfgReader = @Video.TrackingNet;
    elseif strcmpi(dataset, 'UAV123')
        videos = Dataset.UAV123;
        cfgReader = @Video.UAV123;
    else
        videos = Dataset.OTBAttr(dataset);
        cfgReader = @Video.OTB;
    end

end

