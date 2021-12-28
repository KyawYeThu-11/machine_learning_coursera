function compressImage(filename, K, max_iters)
%COMPRESSIMAGE compresses the given image by running Kmeans algorithm 

if ~exist('filename','var') && ~exist('K','var') && ~exist('max_iters','var')
    filename = input('Enter the name of the image file:','s');
    K = input('With how many colors do you want to compress?');
    max_iters = 10;
end

%  Load the input image file
A = double(imread(filename));
A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels. 
% Each row will contain RGB pixel values.
X = reshape(A, img_size(1) * img_size(2), 3);

initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, ~] = runkMeans(X, initial_centroids, max_iters);

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% Recover the image from the indices (idx) by mapping each pixel 
% (specified by it's index in idx) to the centroid value.
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
figure;
subplot(1, 2, 1);
imagesc(A); 
title('Original');
axis square

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
axis square
end

