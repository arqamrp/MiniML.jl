# SCALAR TENSOR OPERATIONS
# ["^", "*", "+", "-", "/", "isless"]

import Base: ^
function ^(arg1::Tensor, arg2::Number)
    """
    ^(arg1::Tensor, arg2::Number)

    Raises each element of the tensor to the power of a scalar value.

    # Arguments
    - `arg1::Tensor`: The tensor to be raised to a power.
    - `arg2::Number`: The scalar exponent.

    # Returns
    - `Tensor`: A new tensor representing the result of the exponentiation.
    """
    ans = arg1.data .^ arg2
    grad_fn = nothing
    if arg1.requires_grad
        grad_fn = (op_res) -> begin
            x = op_res.prev[1].data
            n = arg2
            op_res.prev[1].grad .= op_res.prev[1].grad .+ (n * x.^(n - 1) .* op_res.grad)
        end
    end
    return Tensor(data = ans, prev = (arg1,arg2), op = "^", requires_grad = arg1.requires_grad, grad_fn = grad_fn)
end

import Base: *
function *(arg1::Tensor, arg2::Number)
    """
    *(arg1::Tensor, arg2::Number)

    Multiplies each element of the tensor by a scalar value.

    # Arguments
    - `arg1::Tensor`: The tensor to be multiplied.
    - `arg2::Number`: The scalar multiplier.

    # Returns
    - `Tensor`: A new tensor representing the result of the scalar multiplication.
    """
    val = arg1.data .* arg2
    grad_fn = nothing
    if arg1.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad .= op_res.prev[1].grad .+ (arg2 .* op_res.grad)
        end
    end
    return Tensor(data = val, prev = (arg1, arg2), op = "*", requires_grad = arg1.requires_grad, grad_fn = grad_fn)
end

function *(s::Number, t::Tensor)
    """
    *(arg1::Number, arg2::Tensor)

    Multiplies a scalar value by each element of the tensor.

    # Arguments
    - `arg1::Number`: The scalar multiplier.
    - `arg2::Tensor`: The tensor to be multiplied.

    # Returns
    - `Tensor`: A new tensor representing the result of the scalar multiplication.
    """
    return *(t, s)
end

import Base: /
function /(arg1::Number, arg2::Tensor)
    """
    /(arg1::Number, arg2::Tensor)

    Divides a scalar by a tensor elementwise.

    # Arguments
    - `arg1::Number`: The scalar value to be divided.
    - `arg2::Tensor`: The tensor that divides the scalar.

    # Returns
    - `Tensor`: A new tensor representing the result of the division.
    """
    val = arg1 ./ arg2.data
    grad_fn = nothing
    if arg2.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[2].grad .= op_res.prev[2].grad .- (op_res.grad .* (arg1 ./ (op_res.prev[2].data .^ 2)))
        end
    end
    return Tensor(data = val, prev = (arg1, arg2), op = "/", requires_grad = arg2.requires_grad, grad_fn = grad_fn)
end

function /(arg1::Tensor, arg2::Number)
    """
    /(arg1::Tensor, arg2::Number)

    Divides each element of the tensor by a scalar value.

    # Arguments
    - `arg1::Tensor`: The tensor to be divided.
    - `arg2::Number`: The scalar divisor.

    # Returns
    - `Tensor`: A new tensor representing the result of the division.
    """
    inv_arg2 = 1 / arg2
    return arg1 * inv_arg2
end

# Scalar addition (tensor + k)
import Base:+
function +(arg1::Tensor, arg2::Number)
    """
    +(arg1::Tensor, arg2::Number)

    Adds a scalar value to each element of the tensor.

    # Arguments
    - `arg1::Tensor`: The tensor to which the scalar is added.
    - `arg2::Number`: The scalar value to be added.

    # Returns
    - `Tensor`: A new tensor representing the result of the addition.
    """
    val = arg1.data .+ arg2
    grad_fn = nothing
    if arg1.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad .= op_res.prev[1].grad .+ op_res.grad
        end
    end
    return Tensor(data = val, prev = (arg1,arg2), op = "+", requires_grad = arg1.requires_grad, grad_fn = grad_fn)
end

function +(arg1::Number, arg2::Tensor)
    """
    +(arg1::Number, arg2::Tensor)

    Adds a scalar value to each element of the tensor.

    # Arguments
    - `arg1::Number`: The scalar value to be added.
    - `arg2::Tensor`: The tensor to which the scalar is added.

    # Returns
    - `Tensor`: A new tensor representing the result of the addition.
    """
    return +(arg2, arg1)
end

# scalar subtraction (tensor - k)
import Base:-
function -(arg1::Tensor, arg2::Number)
    """
    -(arg1::Tensor, arg2::Number)

    Subtracts a scalar value from each element of the tensor.

    # Arguments
    - `arg1::Tensor`: The tensor from which the scalar is subtracted.
    - `arg2::Number`: The scalar value to be subtracted.

    # Returns
    - `Tensor`: A new tensor representing the result of the subtraction.
    """
    temp = -arg2
    return arg1 + temp
end

function -(arg1::Number, arg2::Tensor)
    """
    -(arg1::Number, arg2::Tensor)

    Subtracts each element of the tensor from a scalar value.

    # Arguments
    - `arg1::Number`: The scalar value.
    - `arg2::Tensor`: The tensor to be subtracted from the scalar.

    # Returns
    - `Tensor`: A new tensor representing the result of the subtraction.
    """
    return arg1 + (-arg2)
end

# Comparison
import Base.isless
function isless(arg1::Tensor, arg2::Number)
    """
    isless(arg1::Tensor, arg2::Number)

    Compares each element of the tensor to a scalar value.

    # Arguments
    - `arg1::Tensor`: The tensor to compare.
    - `arg2::Number`: The scalar value to compare against.

    # Returns
    - `Tensor`: A new tensor with elements as `1` if they are less than the scalar, otherwise `0`.
    """
    val = (arg1.data .< arg2) * 1
    return Tensor(data = val, prev = (arg1,), op = "<", grad_fn = nothing)
end

function isless(arg1::Number, arg2::Tensor)
    """
    isless(arg1::Number, arg2::Tensor)

    Compares a scalar value to each element of the tensor.

    # Arguments
    - `arg1::Number`: The scalar value to compare.
    - `arg2::Tensor`: The tensor to compare against.

    # Returns
    - `Tensor`: A new tensor with elements as `1` if the scalar is less than the tensor element, otherwise `0`.
    """
    val = (arg2.data .> arg1) * 1.
    return Tensor(data = val, prev = (arg2,), op = ">", grad_fn = nothing)
end