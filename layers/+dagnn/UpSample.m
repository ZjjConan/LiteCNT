classdef UpSample < dagnn.Layer
  properties
    scale = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnupsample(inputs{1}, [], 'scale', obj.scale) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnupsample(inputs{1}, derOutputs{1}, 'scale', obj.scale);
      derParams = {} ;
    end

    function obj = UpSample(varargin)
      obj.load(varargin{:}) ;
      obj.scale = obj.scale ;
    end
  end
end
