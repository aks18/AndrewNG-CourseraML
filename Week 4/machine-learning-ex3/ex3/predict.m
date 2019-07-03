function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%for i = 1:m
  %Taking one row from X at a time
 % temp1 = X(i, 1:end);
 % temp1 = temp1';
 % input = [1; temp1]; %adding the bias term 1
  %Multiplying Input with Theta1
 % layer2 = Theta1 * input;
  %Adding bias in the layer2 for multiplication with Theta2
 % layer2 = [1; layer2];
  %Multiplying Layer2 with Theta2
 % output = Theta2 * layer2;
 % [temp_prob_values p(i)] = max(output);
%endfor

temp1 = [ones(m,1) X];  %Adding columns of 1s - temp1(5000x401)
temp2 = temp1'; %Transpose - temp2(401x5000)
%Theta1(25x401) * InputTranspose(401x5000)
layer2 = Theta1 * temp2;  %layer2(25x5000)

layer2 = sigmoid(layer2); %applying sigmoid function after calculating matrix values
%Layer2(25x5000)
temp3 = layer2'; %Transpose - temp3(5000x25)
temp4 = [ones(m,1) temp3];  %Adding column of 1s - temp4(5000x26)
temp5 = temp4'; %Transpose - temp5(26x5000)

%Theta2(10x26) * temp5(26x5000)
output = Theta2 * temp5; %output(10x5000)

[temp_prob_values p_temp] = max(output, [], 1);
p = p_temp';
% =========================================================================


end
