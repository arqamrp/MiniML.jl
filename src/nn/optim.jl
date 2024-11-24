


function gradient_descend!(arg::Tensor; lr::Float64)
    if any(isnan, arg.grad)
        println(arg)
    end
    @assert !any(isnan, arg.grad) "Gradient should be finite"
    arg.data .= arg.data - lr * arg.grad
end

function gd_step!(model::NeuralNet; lr::Float64)
    for (layer_idx, layer) in model.params
        weights, bias = layer
        gradient_descend!(weights; lr = lr)
        gradient_descend!(bias; lr = lr)
    end
end