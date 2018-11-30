classdef FeatrNorm < dagnn.ElementWise
  properties
    scale = 1
  end
     
  methods
    function outputs = forward(obj, inputs, params)
       [r, c, d, n] = size(inputs{1});
       obj.scale = sqrt(r*c*d/sum(reshape(inputs{1}, [], 1, 1, n).^2, 1) + eps);
       outputs{1} = bsxfun(@times, inputs{1}, obj.scale);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = bsxfun(@times, derOutputs{1}, obj.scale);
      derParams = {};
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
    end
    
    function obj = FeatrNorm(varargin)
      obj.load(varargin) ;
      obj.scale = obj.scale;
    end
  end
end
