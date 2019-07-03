function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
positive_data = find(y == 1);
negative_data = find(y == 0);

%plotting X(positive_data, 1) i.e. exam 1 scores for the students who passed on 
%X-axis and X(positive_data, 2) i.e. exam 2 scores for the students who passed
%on the Y-axis. And plotting the point in the graph using a ks - blacK square
plot(X(positive_data, 1), X(positive_data, 2), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 7);

%plotting X(negative_data, 1) i.e. exam 1 scores for the students who failed on 
%X-axis and X(negative_data, 2) i.e. exam 2 scores for the students who failed
%on the Y-axis. And plotting the point in the graph using a ro - red circle
plot(X(negative_data, 1), X(negative_data, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7); 
% =========================================================================

hold off;

end
