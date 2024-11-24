mutable struct NeuralNet
    layer_dims::Tuple{Vararg{Integer}}
    params::Dict{Integer, Tuple{Vararg{Tensor}}}
    activation_fn::Function
    postprocess_fn::Union{Function, Nothing}

    function NeuralNet(layer_dims::Tuple{Vararg{Integer}}, params::Dict{Integer, Tuple{Vararg{Tensor}}}, activation_fn::Function, postprocess_fn::Union{Function, Nothing})
        new(layer_dims, params, activation_fn, postprocess_fn)
    end
end


function NeuralNet(layer_dims::Tuple{Vararg{Integer}}; activation_fn::Function, postprocess_fn::Union{Function, Nothing} = nothing)
    init_params = Dict{Integer, Tuple{Vararg{Tensor}}}()
    for i in range(1,length(layer_dims)-1)
        W = randn(Float64, (layer_dims[i+1], layer_dims[i]))
        b = randn(Float64, (layer_dims[i+1], 1))
        W_tensor = tensor(W; requires_grad = true, trainable = true)
        b_tensor = tensor(b; requires_grad = true, trainable = true)
        init_params[i] = (W_tensor, b_tensor)
    end
    return NeuralNet(layer_dims, init_params, activation_fn, postprocess_fn)
end

function forward(model::NeuralNet, data::Tensor; postprocess::Bool = true)
    x = data
    num_layers = length(model.layer_dims) 
    for i in range(1, num_layers - 1)
        params = model.params[i]
        W, b = params
        x = W * x + b
        if i != num_layers - 1
            x = model.activation_fn(x)
        end
    end
    if model.postprocess_fn !== nothing && postprocess == true
        x = model.postprocess_fn(x)
    end
    return x
end

function train!(;model::NeuralNet, loss_fn::Function, X::Tensor, y::Tensor, lr::Float64=0.001, num_epochs::Integer = 10)
    losses = zeros((num_epochs,))
    for i in 1:num_epochs
        pred = forward(model, X)
        # println(pred.shape)
        
        loss = loss_fn(y, pred)
        # println(loss.shape)
        losses[i] = loss.data[]
        backward!(loss)
        gd_step!(model, lr=lr)

        zerograd!(loss)
    end
    return losses
end


function batch_train!(;model::NeuralNet, loss_fn::Function, X_batches::Vector{Tensor}, y_batches::Vector{Tensor}, lr::Float64=0.001, num_epochs::Integer = 10)
    losses = zeros((num_epochs,))
    for epoch_idx in 1:num_epochs
        for batch_idx in 1:length(X_batches)
            X = X_batches[batch_idx]
            y = y_batches[batch_idx]
            
            pred = forward(model, X, postprocess = true)
            loss = loss_fn(pred, y)
            backward!(loss)
            gd_step!(model, lr=lr)
    
            zerograd!(loss)
            losses[epoch_idx] += loss.data[]
        end
        losses[epoch_idx] = losses[epoch_idx]/length(X_batches) # average loss for one epoch assuming equal batch sizes throughout
        println("Epoch $(epoch_idx)")
        println(losses[epoch_idx])
    end
    return losses
end