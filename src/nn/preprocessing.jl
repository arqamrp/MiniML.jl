function make_batches(X::AbstractArray, y::AbstractArray, batch_size::Integer)
    n_samples = size(X)[2]
    shuffled_indices = randperm(n_samples)
    X_shuffled = X[:, shuffled_indices]
    y_shuffled = y[shuffled_indices]

    n_batches = div(n_samples, batch_size)
    X_batches = Vector{Tensor}(undef, n_batches)
    y_batches = Vector{Tensor}(undef, n_batches)
    
    for i in 1:n_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = i * batch_size
        X_batches[i] = tensor(X_shuffled[:, start_idx:end_idx], requires_grad= false)
        y_batches[i] = onehot(y_shuffled[start_idx:end_idx], num_classes = 10)
    end

    return (X_batches, y_batches)
end