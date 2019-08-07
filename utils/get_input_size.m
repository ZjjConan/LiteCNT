function inSize = get_input_size(net, ouSize)

    inSize = ouSize;
    for i = numel(net.layers):-1:1
        block = net.layers(i).block;
        if isa(block, 'dagnn.Conv')
            inSize(1) = (inSize(1) - 1) * block.stride(2) - sum(block.pad(1:2)) ...
                + block.size(2);
            inSize(2) = (inSize(2) - 1) * block.stride(1) - sum(block.pad(3:4)) ...
                + block.size(1);
        elseif isa(block, 'dagnn.Pooling')
            inSize(1) = (inSize(1) - 1) * block.stride(2) - sum(block.pad(1:2)) ...
                + block.poolSize(1);
            inSize(2) = (inSize(2) - 1) * block.stride(1) - sum(block.pad(3:4)) ...
                + block.poolSize(1);
        end  
    end

end

