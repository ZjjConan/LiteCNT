classdef LossShrikage < dagnn.Loss

  properties
    a = 10
    c = 0.2
  end  
    
  methods
    function outputs = forward(obj, inputs, params)
      v = exp(1.6*inputs{2});
      lab = inputs{2} .* v;
      lab = lab ./ max(lab(:));
      d = abs(inputs{1} - inputs{2}); 
        
      a1 = lab.^2;
      a2 = d.^2;
      a3 = 1.0 ./ (1 + exp(obj.a .* (obj.c - d)));
      loss = a1 .* a2 .* a3;
      outputs{1} = sum(abs(loss(:)));
      obj.accumulateAverage(inputs, outputs) ;
      
%         label_exp = exp(1.6*label);
% labels = label_exp.*label;
% labels = labels./max(labels(:));
% diff = abs(pred - label);
% a  = 10; 
% c = 0.2;
% a1 =labels.^2;
% a2 = diff.^2;
% a3 = 1.0./(1+exp(a.*(c-abs(diff))));
% loss =a1.*a2.*a3;
% delta = -labels.^2.*(2.*diff./(exp(a.*(c-diff))+1)+ ...
%      a.*diff.^2.*exp(a.*(c-diff))./((exp(a.*(c-diff))+1).^2));
 %delta(delta>0)=0;
    end

    function accumulateAverage(obj, inputs, outputs)
      if obj.ignoreAverage, return; end;
      n = obj.numAveraged ;
      m = n + numel(inputs{1}) ;
      obj.average = outputs{1};
%       obj.average = sqrt(bsxfun(@plus, n * obj.average.^2, gather(outputs{1})) / m) ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      v = exp(1.6*inputs{2});
      lab = inputs{2} .* v;
      lab = lab ./ max(lab(:));
      d = abs(inputs{1} - inputs{2});  
              
      derInputs{1} = ...
          -lab.^2 .* (2.* d ./ (exp(obj.a .* (obj.c - d)) + 1) + ...
          obj.a .* d.^2 .* exp(obj.a .* (obj.c-d)) ./ ((exp(obj.a.*(obj.c - d))+1).^2)) * derOutputs{1};
      
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

    function obj = LossShrikage(varargin)
      obj.load(varargin) ;
    end
  end
end
