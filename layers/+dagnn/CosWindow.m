classdef CosWindow < dagnn.ElementWise
  properties
    window = 1;
  end
     
  methods
    function outputs = forward(obj, inputs, params)
       outputs{1} = bsxfun(@times, inputs{1}, obj.window);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = bsxfun(@times, derOutputs{1}, obj.window);
      derParams = {};
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
    end
    
    function obj = CosWindow(varargin)
      obj.load(varargin) ;
      obj.window = obj.window;
    end
  end
end
