function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

sumVec = sum(X, 1);
sumVec = (1/m) * sumVec;
mu = sumVec';

sumVec = repmat(sumVec, m, 1);  %making sumVec of mxn dimensions
meanDiffMatrix = X - sumVec;
meanDiffMatrix = meanDiffMatrix .^ 2;
meanDiffVec = sum(meanDiffMatrix, 1);  %summation of columns
meanDiffVec = (1/m) * meanDiffVec;
sigma2 = meanDiffVec';

% =============================================================


end
