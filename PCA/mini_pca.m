clc
N = 100
k = 3;
figure;
theta = (2*pi*(1/N:1/N:1))';


x = [cos(theta) sin(theta) cos(k*theta) sin(k*theta)] *(1/(1+k^2))

scatter3(x(:,1),x(:,2),x(:,3))

[X, reconstruc, s,U,mu]  = PCA(x',2);


figure; 
hold on;
plot(cos(theta),sin(theta),'-k','LineWidth',3)
plot(X(1,:),X(2,:),'-r','LineWidth',2)
[X, reconstruc2, s2,U2,mu2]  = PCA(x',3);


figure; 
scatter3(X(1,:),X(2,:),X(3,:));

