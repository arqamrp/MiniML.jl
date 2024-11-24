# ACCESS OPERATIONS
# ["getindex", "reshape", "broadcast", "sum", "mean"]

# Indexing (tensor[idx])
import Base: getindex
function getindex(arg::Tensor, idx...)
    """
    getindex(arg::Tensor, idx...)

    Returns the tensor for the provided indices.

    # Arguments
    - `arg::Tensor`: The tensor to index.
    - `idx...`: The indices for indexing.

    # Returns
    - `Tensor`: A new tensor representing the indexed data.
    """
    val = arg.data[idx...]
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad[idx...] += op_res.grad
        end
    end
    return Tensor(data = val, prev = (arg, (idx...)), op = "getindex", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end

# RESHAPING OPERATIONS
import Base:reshape
function reshape(arg::Tensor, newshape::Tuple{Int, Vararg{Int}})
    """
    reshape(arg::Tensor, newshape::Tuple)

    Reshapes the tensor to anew shape.

    # Arguments
    - `arg::Tensor`: The tensor to reshape.
    - `newshape::Tuple`: The target shape.

    # Returns
    - `Tensor`: A new reshaped tensor.
    """
    arr = arg.data
    shp = arg.shape
    val = reshape(arg.data, newshape)
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (opres)-> begin
            opres.prev[1].grad = reshape(opres.grad, opres.prev[2])
        end
    end
    return Tensor(data = val, prev = (arg, shp), op = "reshape", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end

# Naive broadcasting along 1 dimension
function broadcast( arg::Tensor, newshape::Tuple{Int, Vararg{Int}}) # only from n-1 to n dimensions
    """
    broadcast(arg::Tensor, newshape::Tuple)

    Broadcasts the tensor along a new shape, expanding dimensions where needed.

    # Arguments
    - `arg::Tensor`: The tensor to broadcast.
    - `newshape::Tuple`: The target shape for broadcasting.

    # Returns
    - `Tensor`: A new tensor representing the broadcasted data.
    """
    arr = arg.data
    shp = arg.shape
    
    # check compaitibility
    @assert length(shp) == length(newshape) "Shape mismatch: Dimensions must match"
    for i in 1:length(shp)
        @assert shp[i] == newshape[i] || shp[i] == 1 "Broadcasting incompatible at dimension $i"
    end

    idxs = []
    for i in 1:length(shp)
        if shp[i] != newshape[i] && shp[i] == 1
            push!(idxs, i)
        end
    end

    rep = ones(Int, length(newshape))
    for idx in idxs
        rep[idx] = newshape[idx]
    end
    idxs = Tuple(idxs)

    arr_new = repeat(arr, outer= rep)
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (op_res) -> begin
            op_res.prev[1].grad += sum(op_res.grad, dims = idxs)
        end
    end
    
    return Tensor(data = arr_new, prev = (arg, shp, idxs), op = "broadcast", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end


# AGGREGATION

# Sum along some dimension
import Base:sum
function sum(arg::Tensor; dims::Integer)
    """
    sum(arg::Tensor, dims::Integer)

    Computes the sum of the tensor along a specific dimension.

    # Arguments
    - `arg::Tensor`: The tensor to sum.
    - `dims::Integer`: The dimension along which to compute the sum.

    # Returns
    - `Tensor`: A new tensor representing the summed data.
    """
    # for now along one dimension at a time
    val = sum(arg.data; dims = dims)
    grad_fn = nothing
    if arg.requires_grad
        grad_fn = (op_res) -> begin
            rep = ones(Int, length(op_res.shape))
            rep[dims] = op_res.prev[1].shape[dims]
            op_res.prev[1].grad += repeat(op_res.grad, outer = rep)
        end
    end
    return Tensor(data = val, prev = (arg, dims), op = "sum", requires_grad = arg.requires_grad, grad_fn = grad_fn)
end

# Mean along some dimension

function mean(arg::Tensor;dims::Integer)
    """
    mean(arg::Tensor, dim::Integer)

    Computes the mean of the tensor along a specific dimension.

    # Arguments
    - `arg::Tensor`: The tensor to average.
    - `dim::Integer`: The dimension along which to compute the mean.

    # Returns
    - `Tensor`: A new tensor representing the mean of the data along the specified dimension.
    """
    arr2 = arg/size(arg)[dims]
    tensor_mean = sum(arr2; dims = dims)
    return tensor_mean
end

