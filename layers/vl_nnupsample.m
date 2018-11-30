function y = vl_nnupsample(x, dzdy, varargin)

  opts.scale = 1 ;
  opts = vl_argparse(opts, varargin) ;

  % determine output size
  if opts.scale == 1
    if isempty(dzdy)
      y = x;
    else
      y = dzdy;
    end
    return;
  end
  inSize = [size(x, 1) size(x, 2)] ;
  ouSize = round(inSize * opts.scale);
  
  % generate sampling grid (should probably cache this)
  useGPU = isa(x, 'gpuArray') ;
  Ho = ouSize(1) ; Wo = ouSize(2) ;
  xi = linspace(-1, 1, Ho) ; yi = linspace(-1, 1, Wo) ;
  [yy, xx] = meshgrid(single(xi), single(yi)) ;
  xxyy = [yy(:), xx(:)] ;
  if useGPU, xxyy = gpuArray(xxyy) ; end
  grid = reshape(xxyy, Wo, Ho, 2) ;
  grid = permute(grid, [3,2,1]) ;
  grid = repmat(grid, [1 1 1 size(x, 4)]) ;

 if isempty(dzdy)
   y = vl_nnbilinearsampler(x, grid) ;
 else
   y = vl_nnbilinearsampler(x, grid, dzdy) ;
 end
