# struct for Tensor (instead of class)
mutable struct Tensor{T <: Union{Number, Bool}}
    """
    A struct representing a tensor for automatic differentiation.

    # Fields
    - `shape::Tuple`: The shape of the tensor.
    - `dtype::Type{T}`: The data type of the tensor elements.
    - `data::Union{T, AbstractArray{T}}`: The data stored in the tensor, either as a number or an array.
    - `grad::Union{T, AbstractArray{T}, Nothing}`: The gradient of the tensor, which can be a number, an array, or `nothing`.
    - `requires_grad::Bool`: Indicates whether the tensor requires gradient tracking.
    - `trainable::Bool`: Indicates if the tensor should be modified during training.
    - `prev::Union{Tuple{Vararg{Union{Number, Tensor, Tuple}}}, Tuple{}}`: Stores the previous nodes in the computation graph (or empty tuple).
    - `op::String`: Stores the operation that led to this tensor.
    - `grad_fn::Union{Nothing, Function}`: Stores the gradient function.
    """
    shape::Tuple
    dtype::Type{T}
    data::Union{T, AbstractArray{T}}
    grad::Union{T, AbstractArray{T}, Nothing}
    requires_grad::Bool
    trainable::Bool
    prev::Union{Tuple{Vararg{Union{Number, Tensor, Tuple}}}, Tuple{}}
    op::String
    grad_fn::Union{Nothing, Function}

    function Tensor(shape::Tuple, dtype::Type{T}, data::Union{T, AbstractArray{T}}, grad::Union{T, AbstractArray{T}, Nothing}, requires_grad::Bool, trainable::Bool, prev::Union{Tuple{Vararg{Union{Number, Tensor, Tuple}}}, Tuple{}}, op::String, grad_fn::Union{Nothing, Function}) where {T <: Union{Number, Bool}}
        new{T}(shape, dtype, data, grad, requires_grad, trainable, prev, op, grad_fn)
    end
end

# Non keyword constructor for leaf nodes (numbers or arrays)
function tensor(data::Union{T, AbstractArray{T}}; requires_grad::Bool = true, trainable::Bool = false) where {T <: Union{Number, Bool}}
    """
    Tensor(data::Union{Number, AbstractArray{<:Number}}, requires_grad::Bool=true)

    Creates a leaf node tensor (not arising from a previous tensor) from a number or array. By default, `requires_grad` is set to `true`.

    # Arguments
    - `data::Union{T, AbstractArray{<:T}}`: The initial data for the tensor.
    - `requires_grad::Bool`: Specifies if the tensor should track gradients.
    """
    data_type = eltype(data)
    shape = isa(data, Number) ? () : size(data)
    grad = isa(data, Number) ? zero(data) : zeros(data_type, shape)
    return Tensor(shape, data_type, data, grad, requires_grad, trainable, (), "", nothing)
end

# Constructor for operation results (non-leaf nodes)
function Tensor(; data::Union{T, AbstractArray{T}}, prev::Union{Tuple{Vararg{Union{Number, Tensor, Tuple}}}, Tuple{}}, op::String, grad_fn::Union{Nothing, Function}, requires_grad = true) where {T <: Union{Number, Bool}}
    """
    Tensor(data::Union{Number, AbstractArray{<:Number}}, prev::Tuple{Tensor, Vararg{Tensor}}, op::String)

    Creates a non-leaf node tensor resulting from an operation, storing references to the previous nodes and the operation that led to it.

    # Arguments
    - `data::Union{T, AbstractArray{<:T}}`: The resulting data of the operation.
    - `prev::Union{Tuple{Tensor, Vararg{Tensor}}, Tuple{}}`: The input tensors for the operation.
    - `op::String`: A string pointing to the operation that produced this tensor.
    """
    data_type = eltype(data)
    shape = isa(data, Number) ? () : size(data)
    grad = isa(data, Number) ? zero(data) : zeros(data_type, shape)
    return Tensor(shape, data_type, data, grad, requires_grad, false, prev, op, grad_fn)
end
