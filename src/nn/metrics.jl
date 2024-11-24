# MODEL METRICS


# Classification accuracy
function classification_accuracy(model::NeuralNet, X_test::Tensor, y_test::Tensor)
    preds = forward(model, X_test, postprocess = true) 
    preds = preds.data
    preds = argmax(preds; dims = 1)
    preds = [preds[i].I[1] for i in eachindex(preds)]
    preds = vec(preds)
    return sum(preds.-1 .== y_test.data)/length(y_test.data)
end

# F1
# Precision

# Recall

function mean_squared_error(model::NeuralNet, X_test::Tensor, y_test::Tensor)
    y_pred = forward(model, X_test)
    se = (y_test.data - y_pred.data)^2
    mse = mean(se, dims = 2)
    return mse
end