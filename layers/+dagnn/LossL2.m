classdef LossL2 < dagnn.Loss
    
   properties
    instanceWeights = []
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      delta = inputs{1} - inputs{2} ;
      if ~isempty(obj.instanceWeights)
        delta = delta .* obj.instanceWeights;
      end
      outputs{1} = sum(delta(:).^2) ;
      obj.accumulateAverage(inputs, outputs) ;
    end

    function accumulateAverage(obj, inputs, outputs)
      if obj.ignoreAverage, return; end;
      n = obj.numAveraged ;
      m = n + numel(inputs{1}) ;
      obj.average = sqrt(bsxfun(@plus, n * obj.average.^2, gather(outputs{1})) / m) ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      delta = inputs{1} - inputs{2} ;
      if ~isempty(obj.instanceWeights)
        delta = delta .* obj.instanceWeights;
      end
      derInputs{1} = derOutputs{1} * delta;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = LossL2(varargin)
      obj.load(varargin) ;
    end
  end
end
