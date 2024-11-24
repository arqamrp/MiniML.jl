# BINARY OPERATIONS
# ["+", "-", "*", "**, "/", ".>", ".<", ".==", ".>=", ".<="]

# Tensor addition
import Base: +
function +(arg1::Tensor, arg2::Tensor)
    """
    +(arg1::Tensor, arg2::Tensor)

    Calculates the element-wise sum of two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor representing the sum of the two input tensors.
    """
    if arg1.shape == arg2.shape
        val = arg1.data .+ arg2.data
    else
        if sum(arg2.shape) > sum(arg1.shape) #placeholder logic to replace soon
            return arg2 + broadcast(arg1, arg2.shape)
        else 
            return arg1 + broadcast(arg2, arg1.shape)
        end
    end
    requires_grad = arg1.requires_grad || arg2.requires_grad
    grad_fn = nothing
    if requires_grad
        grad_fn = (op_res) -> begin
            if op_res.prev[1].requires_grad
                op_res.prev[1].grad += op_res.grad
            end
            if op_res.prev[2].requires_grad
                op_res.prev[2].grad += op_res.grad
            end
        end
    end
    return Tensor(data = val, prev = (arg1, arg2), op = "+", requires_grad = requires_grad, grad_fn = grad_fn)
end

# Tensor subtraction
import Base: -
function -(arg1::Tensor, arg2::Tensor)
    """
    -(arg1::Tensor, arg2::Tensor)

    Calculates the element-wise difference of two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor (to subtract from).
    - `arg2::Tensor`: The second tensor (to subtract).

    # Returns
    - `Tensor`: A new tensor representing difference of the two input tensors.
    """
    return arg1 + (-arg2)
end

# Matrix multiplication
import Base: *
function *(arg1::Tensor, arg2::Tensor)
    """
    *(arg1::Tensor, arg2::Tensor)

    Calculates the matrix product of two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor representing the matrix product.
    """
    @assert (ndims(arg1.data) == 2 && ndims(arg2.data) == 2) # vanilla matmult
    val = arg1.data * arg2.data
    
    # add batched matmult later
    # else
    #     A = arg1.data; B = arg2.data
    #     C = Array{eltype(A)}(undef, size(A, 1), size(B)[2:end]...)
    #     Threads.@threads for I in CartesianIndices(axes(A)[3:end])
    #         @views C[:, :, Tuple(I)...] = A[:, :, Tuple(I)...] * B[:, :, Tuple(I)...]
    #     end
    #     return Tensor(data = C, prev = (arg1, arg2), op = '*', grad_fn = grad_fn)
    # end
    requires_grad = arg1.requires_grad || arg2.requires_grad
    grad_fn = nothing
    if requires_grad
        grad_fn = (op_res) -> begin
            A = arg1.data
            B = arg2.data
            if op_res.prev[1].requires_grad
                op_res.prev[1].grad += op_res.grad * B'
            end
            if op_res.prev[2].requires_grad
                op_res.prev[2].grad += A' * op_res.grad
            end
        end
    end

    return Tensor(data = val, prev = (arg1, arg2), op = "*", requires_grad = requires_grad, grad_fn= grad_fn)
end

# Element wise multiplication
import Base: |
function |(arg1::Tensor, arg2::Tensor)
    """
    .*(arg1::Tensor, arg2::Tensor)

    Calculates the element-wise product of two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor representing the elementwise product of the two input tensors.
    """
    @assert arg1.shape == arg2.shape
    val = arg1.data .* arg2.data
    requires_grad = arg1.requires_grad || arg2.requires_grad

    grad_fn = nothing
    if requires_grad
        grad_fn = (op_res) -> begin
            if op_res.prev[1].requires_grad
                op_res.prev[1].grad += op_res.prev[2].data .* op_res.grad
            end
            if op_res.prev[2].requires_grad
                op_res.prev[2].grad += op_res.prev[1].data .* op_res.grad
            end
        end
    end
    return Tensor(data = val, prev = (arg1, arg2), op = "|", requires_grad = requires_grad, grad_fn = grad_fn)
end

# Element wise division
import Base: /
function /(arg1::Tensor, arg2::Tensor)
    """
    /(arg1::Tensor, arg2::Tensor)

    Conducts the element-wise division of a tensor by another.

    # Arguments
    - `arg1::Tensor`: The first tensor (being divided).
    - `arg2::Tensor`: The second tensor (the one dividing).

    # Returns
    - `Tensor`: A new tensor representing the elementwise ratio of the two input tensors.
    """
    val = arg1.data ./ arg2.data
    grad_fn = nothing
    requires_grad = arg1.requires_grad || arg2.requires_grad
    if requires_grad
        grad_fn = (op_res) -> begin
            if op_res.prev[1].requires_grad
                op_res.prev[1].grad +=  op_res.grad./op_res.prev[2].data
            end
            if op_res.prev[2].requires_grad
                op_res.prev[2].grad += -1*op_res.grad .* op_res.prev[1].data ./(op_res.prev[2].data).^2
            end
        end
    end
    return Tensor(data = val, prev = (arg1, arg2), op = "/", requires_grad = requires_grad, grad_fn = grad_fn)
end

# Element wise comparison - not differentiable

# Greater than (>)
import Base:>
function >(arg1::Tensor, arg2::Tensor)
    """
    >(arg1::Tensor, arg2::Tensor)

    Conducts an element-wise > comparison between two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor of Bools representing the elementwise > comparison of the two input tensors.
    """
    val = arg1.data .> arg2.data
    return Tensor(data = val, prev = (arg1, arg2), op = ">", requires_grad = false, grad_fn = nothing)
end

# Less than (<)
import Base: <
function <(arg1::Tensor, arg2::Tensor)
    """
    <(arg1::Tensor, arg2::Tensor)

    Conducts an element-wise < comparison between two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor of Bools representing the elementwise < comparison of the two input tensors.
    """
    val = arg1.data .< arg2.data
    return Tensor(data = val, prev = (arg1, arg2), op = "<", requires_grad = false, grad_fn = nothing)
end

# Equal (==)
import Base: ==
function ==(arg1::Tensor, arg2::Tensor)
    """
    ==(arg1::Tensor, arg2::Tensor)

    Conducts an element-wise equality comparison between two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor of Bools representing the elementwise == comparison of the two input tensors.
    """
    val = arg1.data .== arg2.data
    return Tensor(data = val, prev = (arg1, arg2), op = "==", requires_grad = false, grad_fn = nothing)
end

# Greater than or equal to (>=)
import Base:>=
function >=(arg1::Tensor, arg2::Tensor)
    """
    >=(arg1::Tensor, arg2::Tensor)

    Conducts an element-wise >= comparison between two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor of Bools representing the elementwise >= comparison of the two input tensors.
    """
    val = arg1.data .>= arg2.data
    return Tensor(data = val, prev = (arg1, arg2), op = ">=", requires_grad = false, grad_fn = nothing)
end

# Less than or equal to (<=)
import Base: <=
function <=(arg1::Tensor, arg2::Tensor)
    """
    <(arg1::Tensor, arg2::Tensor)

    Conducts an element-wise <= comparison between two tensors.

    # Arguments
    - `arg1::Tensor`: The first tensor.
    - `arg2::Tensor`: The second tensor.

    # Returns
    - `Tensor`: A new tensor of Bools representing the elementwise <= comparison of the two input tensors.
    """
    val = arg1.data .<= arg2.data
    return Tensor(data = val, prev = (arg1, arg2), op = "<=", requires_grad = false, grad_fn = nothing)
end
