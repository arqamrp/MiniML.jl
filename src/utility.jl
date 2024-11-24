# UTILITY FUNCTIONS
# ["isTensor", "bool_to_num", "size", "isequal"]

function isTensor(arg)
    """
    isTensor(arg)

    Checks if the given argument is an instance of the `Tensor` type.

    # Arguments
    - `arg`: The argument to check.

    # Returns
    - `Bool`: `true` if `arg` is a `Tensor`, otherwise `false`.
    """
    return isa(arg, Tensor)
end

# Utility function for converting a Bool tensor to a Number tensor
function bool_to_num(arg::Tensor)
    """
    bool_to_num_tensor(tensor::Tensor{Bool, <:AbstractArray{Bool}})

    Converts a Bool tensor to a Number tensor, where `true` is mapped to `1` and `false` to `0`.

    # Arguments
    - `tensor::Tensor{Bool, <:AbstractArray{Bool}}`: The input tensor with boolean values.

    # Returns
    - `Tensor{Int, <:AbstractArray{Int}}`: A tensor with numerical values (`1` for `true`, `0` for `false`).
    """
    new_data = map(x -> x ? 1 : 0, arg.data)
    return tensor(new_data, requires_grad = false)
end

import Base:size
function size(arg::Tensor)
    """
    size(arg::Tensor)

    Returns the shape of the given tensor.

    # Arguments
    - `arg::Tensor`: The tensor whose shape is to be returned.

    # Returns   
    - `Tuple`: The shape of the tensor.
    """
    return arg.shape
end


import Base:isequal
function isequal(arg1::Tensor, arg2::Tensor)
    return all(arg1.data .== arg2.data)
end