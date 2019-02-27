
clear;clc;close all;

% Read the image
%A=imread('FishScales.jpg');     subsize=40;
%A=imread('ZebraStripes.jpg');   subsize=20;
A=imread('honeycomb.jpg');      subsize=40;
%A=imread('ortonStripes3.jpg');      subsize=15;


% Downsample a part of the image
A = A(1:10:1000,1:10:1000,:);

% Convert to grayscale
A = rgb2gray(A); 

% Convert to double precision (from integer values)
A = double(A);

% Display
figure(1);
imagesc(A);
colormap(gray);

% Break into subimages
subims=GetSubims(A,subsize);

% Convert from integers to double precision
subims = double(subims);

% PCA
k=10;  % number of PCA modes
[coords,reconstructedSubims,s,u]=PCA(subims,k);

% Plot the PCA coordinates
figure(2);plot3(coords(1,:),coords(2,:),coords(3,:),'.');

% View the subimage space in these coordinates
figure(3); L=30; % number of subimages in each direction
plot_flattened_dataset(coords,subims,L);

% Reconstruct the image from the reconstructed subimages
Arecon = ReconstructFromSubims(reconstructedSubims,A,subsize);

% Display the reconstructed image
figure(4);
imagesc(Arecon);
colormap(gray);

figure;
colormap(gray);
for i = 1:16
subplot(4,4,i);
imagesc(reshape(u(:,i),subsize,subsize));
end
figure;
colormap(gray);

% a = A; 
% for i = 1:16
% subplot(4,4,i);
% a = conv2(a,reshape(u(:,i),subsize,subsize),'same');
% imagesc(a);
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

