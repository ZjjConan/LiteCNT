classdef BranchOut < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.

  properties
    numInputs
    dropIndex
    scale
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
      
      if strcmp(obj.net.mode, 'test')
        outputs{1} = inputs{1};
        for k = 2:obj.numInputs
          outputs{1} = outputs{1} + inputs{k};
        end
      else
        obj.dropIndex = randperm(obj.numInputs, 1);
        obj.scale = obj.numInputs / (obj.numInputs - 1);
        if isa(inputs{1}, 'gpuArray')
          outputs{1} = gpuArray.zeros(size(inputs{1}));
        else
          outputs{1} = zeros(size(params{3}));
        end
        
        for i = 1:obj.numInputs
          if i == obj.dropIndex
            outputs{1} = 0 * inputs{i};
          else
            outputs{1} = obj.scale * inputs{i};
          end
        end
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      for k = 1:obj.numInputs
        if k == obj.dropIndex
          derInputs{k} = 0 * derOutputs{1} ;
        else
          derInputs{k} = obj.scale * derOutputs{1} ;
        end
      end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Sum layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = Sum(varargin)
      obj.load(varargin) ;
    end
  end
end
