function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_array = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_array = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

best = 10;
for i = 1:8
    for j = 1:8
        Ci = C_array(i);
        sigmaj = sigma_array(j);
        model= svmTrain(X, y, Ci, @(x1, x2) gaussianKernel(x1, x2, sigmaj)); 
        y_pred = svmPredict(model, Xval);
        err_rate = mean(double(y_pred ~= yval));        
        if err_rate < best
            best = err_rate;
            C = Ci;
            sigma = sigmaj;
        end
    end
end
disp(best)
disp(sigma)
%disp(length(yval))
%disp(length(y_pred))
disp(C)



% =========================================================================

end
