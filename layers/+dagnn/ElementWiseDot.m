classdef ElementWiseDot < dagnn.ElementWise
%   properties
%     size = [0 0 0 0]
%   end
     
  methods
    function outputs = forward(obj, inputs, params)
       outputs{1} = bsxfun(@times, inputs{1}, inputs{2});
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = bsxfun(@times, derOutputs{1}, inputs{2});
      derInputs{2} = bsxfun(@times, derOutputs{1}, inputs{1});
      derParams = {};
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
    end
    
    function obj = ElementWiseDot(varargin)
      obj.load(varargin) ;
%       obj.size = obj.size;
    end
  end
end
