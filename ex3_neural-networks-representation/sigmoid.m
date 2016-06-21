% ====================================================================
%       Machine Learning
%           Andrew Ng
%            Week 4
%
% Multi-class Classification and Neural Networks: Matlab/Octave code
%
% Author: Antonio Giannino
% ====================================================================
function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end
