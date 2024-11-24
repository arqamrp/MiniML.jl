function back!(x::Tensor)
    if x.grad_fn !== nothing
        x.grad_fn(x)
    end
end

function zero!(arg::Tensor)
    arg.grad = isa(arg.data, Number) ? zero(arg.data) : zeros(arg.dtype, arg.shape)
end

function backward!(x::Tensor)
    stack = []
    push!(stack, x)
    x.grad = isa(x.data, Number) ? one(x.data) : ones(x.dtype, x.shape)
    while !isempty(stack)
        current = pop!(stack)
        back!(current)
        if !isempty(current.prev)
            for parent in current.prev
                if isa(parent, Tensor)
                    push!(stack, parent)
                end
            end
        end
    end
end

function zerograd!(x::Tensor)
    stack = []
    push!(stack, x)

    while !isempty(stack)
        current = pop!(stack)
        zero!(current)
        if !isempty(current.prev)
            for parent in current.prev
                if isa(parent, Tensor)
                    push!(stack, parent)
                end
            end
        end
    end
end