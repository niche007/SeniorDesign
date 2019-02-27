function plot_flattened_dataset(xy,M,k)

    % plot_flattened_dataset - nice 2D layout of image according to sampling location xy
    %
    % plot_flattened_dataset(xy,M,k);
    %
    %	M is a matrix of size (a x b x nbr_images), a collection of images of size (a,b)
    %	xy should be of size (nbr_sample x 2)
    %	k is the number of images stiched in each direction.
    %
    %   Copyright (c) 2005 Gabriel Peyré

    if nargin<3
        k = 30;
    end

    if size(xy,2)>size(xy,1)
        xy = xy';
    end

    a = floor(sqrt(size(M,1)));
    b = a;
    n = size(M,2);

    e = 1/(2*k);
    % plot result
    for i=1:2
        xy(:,i) = rescale(xy(:,i), e, 1-e );
    end

    A = zeros(k*b,k*a) + 1; % mmax(M(:));
    for x=1:k
        for y=1:k
            selx = ((x-1)*b)+1:x*b;
            sely = y*a:-1:((y-1)*a)+1;
            % center point
            cx = e + (x-1)/k;
            cy = e + (y-1)/k;
            % distance to center
            d = max( abs( xy(:,1)-cx ), abs( xy(:,2)-cy ) );
            % find closest point
            [v,I] = min(d);
            if v<=e
                %A(selx,sely) = rescale( M(end:-1:1,:,I)' );
                A(selx,sely) = rescale( reshape(M(:,I),a,b)' )*254/256 + 1/256;
            end
        end
    end


    n = size(A,1);
    q = size(A,2);
    xy(:,1) = xy(:,1)*n;
    xy(:,2) = xy(:,2)*q;

    plot_lines = 1;

    hold off;
    imagesc(A');
    %colormap([0 0 0;jet(254);1 1 1]);
    colormap([0 0 0;gray(254);1 1 1]);
    %colormap(jet(256));
    hold on;
    axis tight; axis square; axis off;
    axis xy;
    if plot_lines
        for x=0:b:n
            line([x+.5 x+.5], [0 q+.5],'Color','k');
        end
        for x=0:a:q
            line([0 n+.5], [x+.5 x+.5],'Color','k');
        end
    end
    hold off;

end




function y = rescale(x,a,b)

    % rescale - rescale data in [a,b]
    %
    %   y = rescale(x,a,b);
    %
    %   Copyright (c) 2004 Gabriel PeyrÈ

    if nargin<2
        a = 0;
    end
    if nargin<3
        b = 1;
    end

    m = min(x(:));
    M = max(x(:));

    y = (b-a) * (x-m)/(M-m) + a;

end