function [X,reconstruction, s,U,mu] = PCA(Y,p)
%%% Algorithm 1.1 Principal Component Analysis (PCA)
%%% Inputs: Y  - n-by-N matrix with columns y_i in R^n for i=1,...,N
%           p  - percentage of variance to maintain (or dimension if p>1)
%%% Output: X  - m-by-N matrix with columns x_i in R^m for i=1,...,N
%           s  - m-by-1 vector, contains the percentage of the total variance
%                captured by variables (x_1,...,x_s(i))
%           U  - m-by-n matrix which projects the data Y onto the principal
%                components
%           mu - n-by-1 vector, mean of the data Y

N = size(Y,2);
mu = mean(Y,2); % mu will be n-by-1
Y = Y-repmat(mu,1,N); % center the data
CY = Y*Y'/N;            % observed covariance matrix
[U,L] = eig(CY);        % eigendecomposition
s=diag(L);              % vector of eigenvalues

[s,inds] = sort(s,'descend');   %re-order the eigenvalues
figure(10);plot(s);
U = U(:,inds);                  %re-order the eigenvectors
X=U'*Y;                 % PCA, recover the hidden variables  


totalVariance=sum(s);
percentVariance = s/totalVariance;
cumulativeVariance = cumsum(percentVariance);
s=cumulativeVariance;

    X=X(1:p,:);


reconstruction = U(:,1:p)*X + repmat(mu,1,N); 