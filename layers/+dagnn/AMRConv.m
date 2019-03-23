classdef AMRConv < dagnn.Conv

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      params{1} = bsxfun(@times, params{1}, params{3});
      outputs{1} = vl_nnconv(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end

      params{1} = bsxfun(@times, params{1}, params{3});
      
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
      
      derParams{3} = sum(bsxfun(@times, derParams{1}, params{1}), 3);  
      derParams{1} = bsxfun(@times, derParams{1}, params{3});

    end
    
    
    function obj = AMRConv(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
    end
  end
end
