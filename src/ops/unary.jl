# UNARY OPERATIONS
# ["exp", "log", "log10", "adjoint", "negation"]

# elementwise exponentiation
import Base: exp
function exp(arg::Tensor)
    """
    exp(arg::Tensor)

    Calculates the element-wise exponential of the given tensor.

    # Arguments
    - `arg::Tensor`: The tensor to exponentiate.

    # Returns
    - `Tensor`: A new tensor with the result of the exponential operation.
    """
    val = exp.(arg.data)
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad +=  op_res.data .* op_res.grad
        end
    end
    return Tensor(data = val, prev = (arg, ), op = "exp", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end

# elementwise logarithm (tensor)
import Base:log
function log(arg::Tensor)
    """
    log(arg::Tensor)

    Calculates the element-wise natural logarithm of the given tensor.

    # Arguments
    - `arg::Tensor`: The tensor to take the logarithm of.

    # Returns
    - `Tensor`: A new tensor with the result of the logarithm operation.
    """
    val = log.(arg.data)
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad +=  op_res.grad./ op_res.prev[1].data
        end
    end

    return Tensor(data = val, prev = (arg,), op = "log", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end

import Base:log10
function log10(arg::Tensor)
    """
    log_10(arg::Tensor)

    Calculates the element-wise base-10 logarithm of the given tensor.

    # Arguments
    - `arg::Tensor`: The tensor to take the logarithm of.

    # Returns
    - `Tensor`: A new tensor with the result of the logarithm operation.
    """
    val = log10.(arg.data)
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad +=  op_res.grad./ (op_res.prev[1].data * log(10))
        end
    end
    return Tensor(data = val, prev = (arg,), op = "log10", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end

# transpose(matrix tensor)
import Base:adjoint
function adjoint(arg::Tensor)
    """
    adjoint(arg::Tensor)
    
    Calculates the transpose of a 2D tensor. Equivalent to shorthand arg'

    # Arguments
    - `arg::Tensor`: The tensor to transpose. Must be 2-dimensional.

    # Returns
    - `Tensor`: A new tensor representing the transposed input tensor.
    """
    @assert length(arg.shape) == 2
    val =Matrix(adjoint(arg.data))
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad +=  Matrix(adjoint(op_res.grad))
        end
    end
    return Tensor(data = val, prev = (arg,), op = "adjoint", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end

# Negation
import Base:-
function -(arg::Tensor)
    """
    -(arg::Tensor)

    Negates each element of the tensor.

    # Arguments
    - `arg::Tensor`: The tensor to be negated.

    # Returns
    - `Tensor`: A new tensor with each element negated.
    """
    return -1 * arg
end
