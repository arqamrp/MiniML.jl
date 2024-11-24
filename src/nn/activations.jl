# redesigned for numerical stability
# Sigmoid function
function sigmoid(arg::Tensor)
    """
    sigmoid(arg::Tensor) -> Tensor

    The sigmoid function applies the element-wise transformation:

        σ(x) = 1 / (1 + exp(-x))

    This maps the input values into the range (0, 1). The function is numerically stable to avoid overflow/underflow issues.

    # Arguments
    - `arg::Tensor`: A tensor containing the input values.

    # Returns
    - A tensor where each element has been transformed using the sigmoid function.
    """
    exparr = exp(-arg)  
    ans = 1.0 / (1.0 + exparr) 
    return ans
end

# ReLu function
function relu(arg::Tensor)
    (arg > 0) | (arg)
end

# Softmax function

function softmax(arg::Tensor; dim::Integer=1)
    max_vals = maximum(arg.data, dims=dim)
    arg.data = arg.data .- max_vals # directly modifying because constants don't affect calculation/gradient

    e_arg = exp(arg)
    sums = sum(e_arg, dims = dim)
    bsums = broadcast(sums, size(e_arg))
    return e_arg/bsums
end