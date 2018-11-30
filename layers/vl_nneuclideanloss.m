function y = vl_nneuclideanloss(x, t, varargin)
% VL_NNEUCLIDEANLOSS computes the L2 Loss
%  Y = VL_NNEUCLIDEANLOSS(X, T) computes the Euclidean Loss
%  (also known as the L2 loss) between an N x 1 array of input
%  predictions, X and an N x 1 array of targets, T. The output
%  Y is a scalar value.
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.

  if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
    dzdy = varargin{1} ;
    varargin(1) = [] ;
  else
    dzdy = [] ;
  end

  opts.instanceWeights = ones(size(x)) ;
  opts = vl_argparse(opts, varargin) ;

  % residuals
  res = x - t ;

  if isempty(dzdy)
    resSq = res.^2 ;
    weighted = (1/2) * opts.instanceWeights .* resSq ;
    y = sum(weighted(:)) ;
  else
    y = opts.instanceWeights .* res * dzdy{1} ;
  end