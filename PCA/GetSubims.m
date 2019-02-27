function subims = GetSubims(A,subimSize,skip)

    if (nargin<2)
        subimSize=4;
    end
    if (nargin<3)
        skip=1;
    end
    n=subimSize;        %%% Subimage size
    N=size(A,1);        %%% full image size

    %%%% Collect the indices of the subimages for fast extraction

    nsubs = floor((N-n+1)/skip);
    subinds = zeros(nsubs^2,n^2);
    for i = 1:nsubs
        for j = 1:nsubs
            [X,Y]=meshgrid((1:n)+skip*(i-1),(1:n)+skip*(j-1));
            subinds(nsubs*(i-1) + j,:) = sub2ind([N N],X(:),Y(:));        
        end
    end
    
    subims = A(subinds)';

end

