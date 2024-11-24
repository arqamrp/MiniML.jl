# LOSS FUNCTIONS

# Sum of squared errors
function sum_squared_error(y_pred::Tensor, y_true::Tensor)
    """
    sum_squared_error(y_pred::Tensor, y_true::Tensor)

    Returns the sum of squared errors for eall data points.

    # Arguments
    - `y_pred::Tensor`: The predicted value of the dependent variable.
    - `y_true::Tensor`: The true value of the dependent variable.

    # Returns
    - `Tensor`: The loss as a Tensor.
    """
    diff2 = (y_true - y_pred)^2
    sse = sum(diff2, dims = 2)
    return sse
end

# Binary cross entropy
function binary_cross_entropy(y_probs::Tensor, y_labels::Tensor)
    """
    binary_cross_entropy(y_probs::Tensor, y_labels::Tensor)

    Returns the binary cross entropy loss (negative log likelihood for Bernoulli data) for all data points.

    # Arguments
    - `y_probs::Tensor`: The predicted probabilities of class 1. Its shape should be (1, n_samples) and all elements in (0,1).
    - `y_labels::Tensor`: The ground truth labels. Its shape should be (1, n_samples) and should contain only 0s and 1s.

    # Returns
    - `Tensor`: The loss as a Tensor.
    """
    c = 1e-10
    y_probs.data = clamp.(y_probs.data, c, 1)
    
    losses = -  y_labels | log( y_probs) - (1 - y_labels) | log(1-y_probs) # shape (1, n_samples)
    loss = mean(losses, dims = 2)
    return loss
end

function categorical_cross_entropy(y_probs::Tensor, y_onehot::Tensor)
    """
    categorical_cross_entropy(y_probs::Tensor, y_onehot::Tensor)

    Returns the categorical cross entropy loss (negative log likelihood for categorical distribution data) for all data points.

    # Arguments
    - `y_probs::Tensor`: The predicted probabilities of each class, lying between (0,1). Its shape should be (n_classes, n_samples). The col sums should be 1. 
    - `y_onehot::Tensor`: The ground truth labels, as one hot vectors. Its shape should be (1, n_samples) and should contain only integers in {1,.. n_classes}.

    # Returns
    - `Tensor`: The loss as a Tensor.
    """
    c = 1e-10
    y_probs.data = clamp.(y_probs.data, c, 1)
    
    losses = sum( - y_onehot | log(y_probs) ,  dims = 1)
    loss = mean(losses, dims = 2)
    return loss
end

# function categorical_cross_entropy(logits::Tensor, y_onehot::Tensor)
#     ## logits.shape should be (n_classes, n_samples)
#     ## y_onehot.shape should be (n_classes, n_samples) - should contain one-hot encoded labels

#     max_logits = maximum(logits.data, dims=1)
#     log_sum_exp = max_logits .+ log.(sum(exp.(logits.data .- max_logits), dims=1)) # log sum exp trick
#     losses = sum(y_onehot | (logits - log_sum_exp), dims=1)
#     loss = -mean(losses, dims = 2)

#     return loss
# end