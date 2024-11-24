module MiniML

include("tensor.jl")
include("utility.jl")
include("std_tensors.jl")
include("autograd.jl")

include("./ops/binary.jl")
include("./ops/scalar.jl")
include("./ops/tensor_ops.jl")
include("./ops/unary.jl")

include("./nn/NeuralNet.jl")
include("./nn/activations.jl")
include("./nn/preprocessing.jl")
include("./nn/losses.jl")
include("./nn/metrics.jl")
include("./nn/optim.jl")

export Tensor, tensor, NeuralNet
end
