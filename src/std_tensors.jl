# function onehotvec(label::Integer; num_classes::Integer)
#     # assumes labels from 0 to num_classes-1
#     @assert label <= num_classes
#     nohot = zeros(Int64, (num_classes, 1))
#     nohot[ label + 1, 1] = 1
#     return tensor(nohot, requires_grad = false)
# end


function onehot(labels::Vector{<:Integer}; num_classes::Integer)
    @assert all(labels .>= 0) "Labels must be non-negative."
    @assert all(labels .< num_classes) "Each label must be less than num_classes."
    @assert length(size(labels)) == 1  "Pass labels as a 1D array"

    num_samples = size(labels)[1]
    nohot = zeros(Int64, (num_classes, num_samples) )

    # list of all linear indices that should be set to one: (julia is column first)
    hotten = collect((0:num_samples-1)*(num_classes)) # initialise to the first -1 th index for each column (sample)
    hotten .= hotten .+ (labels.+1) # add label value
    nohot[hotten] .= 1
    return tensor(nohot, requires_grad = false)
end