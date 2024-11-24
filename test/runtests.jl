using MiniML
using Test

@testset "miniML.jl" begin
    @testset "Tensor Constructors" begin
        # Test creation of a scalar tensor with requires_grad = true (default)
        scalar_tensor_default = tensor(10.0)
        @test scalar_tensor_default.data == 10.0
        @test scalar_tensor_default.shape == ()
        @test scalar_tensor_default.requires_grad == true
        @test scalar_tensor_default.dtype == Float64
        @test scalar_tensor_default.grad == 0.0
    
        # Test creation of a scalar tensor with requires_grad = false
        scalar_tensor_no_grad = tensor(5.0, requires_grad = false)
        @test scalar_tensor_no_grad.data == 5.0
        @test scalar_tensor_no_grad.requires_grad == false
        @test scalar_tensor_no_grad.grad == 0.0
    
        # Test creation of a 1D array tensor with requires_grad = true (default)
        array_data_1d = [1.0, 2.0, 3.0]
        array_tensor_1d = tensor(array_data_1d)
        @test array_tensor_1d.data == array_data_1d
        @test array_tensor_1d.shape == (3,)
        @test array_tensor_1d.requires_grad == true
        @test array_tensor_1d.dtype == Float64
        @test array_tensor_1d.grad == zeros(Float64, 3)
    
        # Test creation of a 1D array tensor with requires_grad = false
        array_tensor_1d_no_grad = tensor(array_data_1d, requires_grad = false)
        @test array_tensor_1d_no_grad.data == array_data_1d
        @test array_tensor_1d_no_grad.requires_grad == false
        @test array_tensor_1d_no_grad.grad == zeros(Float64, 3)
    
        # Test creation of a 2D array tensor with requires_grad = true (default)
        array_data_2d = [ 1.0 2.0; 3.0 4.0 ]
        array_tensor_2d = tensor(array_data_2d)
        @test array_tensor_2d.data == array_data_2d
        @test array_tensor_2d.shape == (2, 2)
        @test array_tensor_2d.requires_grad == true
        @test array_tensor_2d.dtype == Float64
        @test array_tensor_2d.grad == zeros(Float64, 2, 2)
    
        # Test adjoint of a 2D array tensor with requires_grad = true (default)
        adjoint_array_tensor_2d = array_tensor_2d'
        @test adjoint_array_tensor_2d.data == array_data_2d'
        @test adjoint_array_tensor_2d.shape == (2, 2)
        @test adjoint_array_tensor_2d.requires_grad == true
        @test adjoint_array_tensor_2d.dtype == Float64
        @test adjoint_array_tensor_2d.grad == zeros(Float64, 2, 2)
    
        # Test creation of a tensor with an integer scalar
        int_scalar_tensor = tensor(7)
        @test int_scalar_tensor.data == 7
        @test int_scalar_tensor.shape == ()
        @test int_scalar_tensor.requires_grad == true
        @test int_scalar_tensor.dtype == Int64
        @test int_scalar_tensor.grad == 0
    
        # Test creation of a tensor with an integer array
        int_array_data = [1, 2, 3]
        int_array_tensor = tensor(int_array_data)
        @test int_array_tensor.data == int_array_data
        @test int_array_tensor.shape == (3,)
        @test int_array_tensor.requires_grad == true
        @test int_array_tensor.dtype == Int64
        @test int_array_tensor.grad == zeros(Int64, 3)
    
        # Test creation of a Float32 tensor
        array_data_float32 = Float32[1.0, 2.0, 3.0]
        array_tensor_float32 = tensor(array_data_float32)
        @test array_tensor_float32.data == array_data_float32
        @test array_tensor_float32.shape == (3,)
        @test array_tensor_float32.requires_grad == true
        @test array_tensor_float32.dtype == Float32
        @test array_tensor_float32.grad == zeros(Float32, 3)
    
        # Test creation of a Boolean tensor
        bool_data = [true, false, true]
        bool_tensor = tensor(bool_data)
        @test bool_tensor.data == bool_data
        @test bool_tensor.shape == (3,)
        @test bool_tensor.requires_grad == true
        @test bool_tensor.dtype == Bool
        @test bool_tensor.grad == zeros(Bool, 3)
    end

    @testset "Utility Functions for Tensor" begin
        # Define some sample tensors for testing
        scalar_tensor = tensor(5.0, requires_grad = true)
        array_tensor = tensor([1.0, 2.0, 3.0], requires_grad = true)
        bool_tensor = tensor([true, false, true], requires_grad = false)
    
        # Test MiniML.isTensor function
        @testset "MiniML.isTensor Function" begin
            @test MiniML.isTensor(scalar_tensor) == true
            @test MiniML.isTensor(array_tensor) == true
            @test MiniML.isTensor(bool_tensor) == true
            @test MiniML.isTensor(42) == false
            @test MiniML.isTensor([1, 2, 3]) == false
            @test MiniML.isTensor("Not a tensor") == false
        end
    
        # Test MiniML.bool_to_num function
        @testset "MiniML.bool_to_num Function" begin
            converted_tensor = MiniML.bool_to_num(bool_tensor)
            @test converted_tensor.data == [1, 0, 1]
            @test converted_tensor.requires_grad == false
            @test converted_tensor.dtype == Int64
            @test converted_tensor.shape == (3,)
        end
    
        # Test size function
        @testset "size Function" begin
            @test size(scalar_tensor) == ()
            @test size(array_tensor) == (3,)
            @test size(bool_tensor) == (3,)
        end
    
        # Test isequal function
        @testset "isequal Function" begin
            # Define tensors with the same data
            tensor1 = tensor([1.0, 2.0, 3.0])
            tensor2 = tensor([1.0, 2.0, 3.0])
            tensor3 = tensor([4.0, 5.0, 6.0])
    
            @test isequal(tensor1, tensor2) == true
            @test isequal(tensor1, tensor3) == false
            @test isequal(scalar_tensor, tensor(5.0)) == true
            @test isequal(scalar_tensor, tensor(10.0)) == false
        end
    end

    
    @testset "Tensor access and aggregation operations" begin
        @testset "getindex operation" begin
            data = [1.0, 2.0, 3.0, 4.0]
            t = tensor(data, requires_grad=true)
            idx = 3
            indexed_tensor = t[idx]
            @test indexed_tensor.data == 3.0
            
            # Test gradient computation
            indexed_tensor.grad = 1.0  # Set gradient of the result tensor
            indexed_tensor.grad_fn(indexed_tensor)
            @test t.grad == [0.0, 0.0, 1.0, 0.0]  # Expect only third element to have a gradient
        end

        # Test `reshape`
        @testset "reshape operation" begin|
            data = [1.0, 2.0, 3.0, 4.0]
            t = tensor(data, requires_grad=true)
            newshape = (2, 2)
            reshaped_tensor = reshape(t, newshape)
            @test reshaped_tensor.shape == newshape
            @test reshaped_tensor.data == reshape(data, newshape)
            
            # Test gradient computation
            reshaped_tensor.grad = [1.0 1.0; 1.0 1.0]  # Set gradient of the reshaped tensor
            reshaped_tensor.grad_fn(reshaped_tensor)
            @test t.grad == [1.0, 1.0, 1.0, 1.0]  # Gradient should be evenly distributed
        end

        # Test `broadcast`
        @testset "broadcast operation" begin
            data = [1.0]
            t = tensor(data, requires_grad=true)
            newshape = (3,)
            broadcasted_tensor = MiniML.broadcast(t, newshape)
            @test broadcasted_tensor.shape == newshape
            @test broadcasted_tensor.data == [1.0, 1.0, 1.0]
            
            # Test gradient computation
            broadcasted_tensor.grad = [1.0, 2.0, 3.0]  # Set gradient of the broadcasted tensor
            broadcasted_tensor.grad_fn(broadcasted_tensor)
            @test t.grad == [6.0]  # The gradient should be the sum of the broadcasted gradients
        end

        # Test `sum` along a dimension
        @testset "sum operation" begin
            data = [1.0 2.0; 3.0 4.0]
            t = tensor(data, requires_grad=true)
            summed_tensor = sum(t, dims=2)
            @test summed_tensor.shape == (2, 1)
            @test summed_tensor.data == [3.0; 7.0;;]
            
            # Test gradient computation
            summed_tensor.grad = [1.0; 1.0]  # Set gradient of the summed tensor
            summed_tensor.grad_fn(summed_tensor)
            @test t.grad == [1.0 1.0; 1.0 1.0]  # Gradient should be evenly distributed along summed dimension
        end

        # Test `mean` along a dimension
        @testset "mean operation" begin
            data = [1.0 2.0; 3.0 4.0]
            t = tensor(data, requires_grad=true)
            mean_tensor = MiniML.mean(t, dims = 2)
            @test mean_tensor.shape == (2, 1)
            @test mean_tensor.data == [1.5; 3.5;;]
            
            # Test gradient computation
            mean_tensor.grad = [1.0; 1.0]  # Set gradient of the mean tensor
            mean_tensor.grad_fn(mean_tensor)
            mean_tensor.prev[1].grad_fn(mean_tensor.prev[1])
            @test t.grad == [0.5 0.5; 0.5 0.5]  # Gradient should be distributed evenly across the averaged values
        end
    end
    
    @testset "Unary Operations for Tensor" begin
        # Define sample tensors for testing
        scalar_tensor = tensor(2.0, requires_grad = true)
        array_tensor = tensor([1.0, 2.0, 3.0], requires_grad = true)
        matrix_tensor = tensor([1.0 2.0; 3.0 4.0], requires_grad = true)
    
        # Test exp function
        @testset "exp Function" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            exp_tensor = exp(array_tensor)
            @test exp_tensor.data == exp.([1.0, 2.0, 3.0])
            @test exp_tensor.requires_grad == true
            @test exp_tensor.shape == (3,)
            @test exp_tensor.op == "exp"
    
            # Set gradient of resulting tensor and manually backpropagate
            exp_tensor.grad = [1.0, 1.0, 1.0]
            if exp_tensor.grad_fn != nothing
                exp_tensor.grad_fn(exp_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == exp.([1.0, 2.0, 3.0])
        end
    
        # Test log function
        @testset "log Function" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            log_tensor = log(array_tensor)
            @test log_tensor.data == log.([1.0, 2.0, 3.0])
            @test log_tensor.requires_grad == true
            @test log_tensor.shape == (3,)
            @test log_tensor.op == "log"
    
            # Set gradient of resulting tensor and manually backpropagate
            log_tensor.grad = [1.0, 1.0, 1.0]
            if log_tensor.grad_fn != nothing
                log_tensor.grad_fn(log_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [1.0, 0.5, 0.3333333333333333]  # New gradient after wipe
        end
    
        # Test log10 function
        @testset "log10 Function" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            log10_tensor = log10(array_tensor)
            @test log10_tensor.data == log10.([1.0, 2.0, 3.0])
            @test log10_tensor.requires_grad == true
            @test log10_tensor.shape == (3,)
            @test log10_tensor.op == "log10"
    
            # Set gradient of resulting tensor and manually backpropagate
            log10_tensor.grad = [1.0, 1.0, 1.0]
            if log10_tensor.grad_fn != nothing
                log10_tensor.grad_fn(log10_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [1.0 / log(10), 0.5 / log(10), 0.3333333333333333 / log(10)]  # New gradient after wipe
        end
    
        # Test adjoint (transpose) function
        @testset "adjoint Function" begin
            MiniML.zerograd!(matrix_tensor)  # Wipe previous grads
            adjoint_tensor = adjoint(matrix_tensor)
            @test adjoint_tensor.data == [1.0 3.0; 2.0 4.0]
            @test adjoint_tensor.requires_grad == true
            @test adjoint_tensor.shape == (2, 2)
            @test adjoint_tensor.op == "adjoint"
    
            # Set gradient of resulting tensor and manually backpropagate
            adjoint_tensor.grad = [1.0 1.0; 1.0 1.0]
            if adjoint_tensor.grad_fn != nothing
                adjoint_tensor.grad_fn(adjoint_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test matrix_tensor.grad == [1.0 1.0; 1.0 1.0]
        end
    
        # Test negation function
        @testset "Negation Function" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            negated_tensor = -array_tensor
            @test negated_tensor.data == [-1.0, -2.0, -3.0]
            @test negated_tensor.requires_grad == true
            @test negated_tensor.shape == (3,)
            @test negated_tensor.op == "*"
    
            # Set gradient of resulting tensor and manually backpropagate
            negated_tensor.grad = [1.0, 1.0, 1.0]
            if negated_tensor.grad_fn != nothing
                negated_tensor.grad_fn(negated_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [-1.0, -1.0, -1.0]  # New gradient after wipe
        end
    end   
    
    @testset "Binary Tensor Operations" begin
        @testset "addition operation" begin
            data1 = [1.0, 2.0, 3.0]
            data2 = [4.0, 5.0, 6.0]
            t1 = tensor(data1, requires_grad=true)
            t2 = tensor(data2, requires_grad=true)
            sum_tensor = t1 + t2
            @test sum_tensor.data == [5.0, 7.0, 9.0]
            
            # Test gradient computation
            sum_tensor.grad = [1.0, 1.0, 1.0]
            sum_tensor.grad_fn(sum_tensor)
            @test t1.grad == [1.0, 1.0, 1.0]
            @test t2.grad == [1.0, 1.0, 1.0]
        end
    
        # Test `-` operation
        @testset "subtraction operation" begin
            data1 = [5.0, 7.0, 9.0]
            data2 = [1.0, 2.0, 3.0]
            t1 = tensor(data1, requires_grad=true)
            t2 = tensor(data2, requires_grad=true)
            diff_tensor = t1 - t2
            @test diff_tensor.data == [4.0, 5.0, 6.0]
            
            # Test gradient computation
            diff_tensor.grad = [1.0, 1.0, 1.0]
            diff_tensor.grad_fn(diff_tensor)
            diff_tensor.prev[2].grad_fn(diff_tensor.prev[2])
            @test t1.grad == [1.0, 1.0, 1.0]
            @test t2.grad == [-1.0, -1.0, -1.0]
        end
    
        # Test `*` operation (Matrix multiplication)
        @testset "matrix multiplication operation" begin
            data1 = [1.0 2.0; 3.0 4.0]
            data2 = [2.0 0.0; 1.0 3.0]
            t1 = tensor(data1, requires_grad=true)
            t2 = tensor(data2, requires_grad=true)
            prod_tensor = t1 * t2
            @test prod_tensor.data == t1.data * t2.data
    
            # Test gradient computation
            prod_tensor.grad = [1.0 1.0; 1.0 1.0]
            prod_tensor.grad_fn(prod_tensor)
            @test t1.grad == [1.0 1.0; 1.0 1.0] * t2.data'
            @test t2.grad == t1.data' * [1.0 1.0; 1.0 1.0]
        end
    
        # Test `|` operation (Element-wise multiplication)
        @testset "element-wise multiplication operation" begin
            data1 = [1.0, 2.0, 3.0]
            data2 = [4.0, 5.0, 6.0]
            t1 = tensor(data1, requires_grad=true)
            t2 = tensor(data2, requires_grad=true)
            mul_tensor = t1 | t2
            @test mul_tensor.data == [4.0, 10.0, 18.0]
            
            # Test gradient computation
            mul_tensor.grad = [1.0, 1.0, 1.0]
            mul_tensor.grad_fn(mul_tensor)
            @test t1.grad == [4.0, 5.0, 6.0]
            @test t2.grad == [1.0, 2.0, 3.0]
        end
    
        # Test `/` operation (Element-wise division)
        @testset "element-wise division operation" begin
            data1 = [8.0, 9.0, 10.0]
            data2 = [2.0, 3.0, 5.0]
            t1 = tensor(data1, requires_grad=true)
            t2 = tensor(data2, requires_grad=true)
            div_tensor = t1 / t2
            @test div_tensor.data == [4.0, 3.0, 2.0]
            
            # Test gradient computation
            div_tensor.grad = [1.0, 1.0, 1.0]
            div_tensor.grad_fn(div_tensor)
            @test t1.grad == [0.5, 0.3333333333333333, 0.2]
            @test t2.grad == [-2.0, -1.0, -0.4]
        end
    
        # Test `>` operation (Element-wise greater than comparison)
        @testset "greater than comparison operation" begin
            data1 = [1.0, 3.0, 5.0]
            data2 = [2.0, 3.0, 4.0]
            t1 = tensor(data1)
            t2 = tensor(data2)
            gt_tensor = t1 > t2
            @test gt_tensor.data == [false, false, true]
        end
    
        # Test `<` operation (Element-wise less than comparison)
        @testset "less than comparison operation" begin
            data1 = [1.0, 3.0, 5.0]
            data2 = [2.0, 3.0, 4.0]
            t1 = tensor(data1)
            t2 = tensor(data2)
            lt_tensor = t1 < t2
            @test lt_tensor.data == [true, false, false]
        end
    
        # Test `==` operation (Element-wise equality comparison)
        @testset "equality comparison operation" begin
            data1 = [1.0, 3.0, 5.0]
            data2 = [1.0, 3.0, 6.0]
            t1 = tensor(data1)
            t2 = tensor(data2)
            eq_tensor = t1 == t2
            @test eq_tensor.data == [true, true, false]
        end
    
        # Test `>=` operation (Element-wise greater than or equal comparison)
        @testset "greater than or equal comparison operation" begin
            data1 = [1.0, 3.0, 5.0]
            data2 = [1.0, 2.0, 5.0]
            t1 = tensor(data1)
            t2 = tensor(data2)
            gte_tensor = t1 >= t2
            @test gte_tensor.data == [true, true, true]
        end
    
        # Test `<=` operation (Element-wise less than or equal comparison)
        @testset "less than or equal comparison operation" begin
            data1 = [1.0, 3.0, 5.0]
            data2 = [1.0, 4.0, 5.0]
            t1 = tensor(data1)
            t2 = tensor(data2)
            lte_tensor = t1 <= t2
            @test lte_tensor.data == [true, true, true]
        end
    end


    @testset "Scalar Tensor operations" begin
        # Define sample tensors for testing
        scalar_tensor = tensor(3.0, requires_grad = true)
        array_tensor = tensor([1.0, 2.0, 3.0], requires_grad = true)
    
        # Test power function (^)
        @testset "Power Function" begin
            power_tensor = array_tensor ^ 2.0
            @test power_tensor.data == [1.0, 4.0, 9.0]
            @test power_tensor.requires_grad == true
            @test power_tensor.shape == (3,)
            @test power_tensor.op == "^"
    
            # Set gradient of resulting tensor and manually backpropagate
            power_tensor.grad = [1.0, 1.0, 1.0]
            if power_tensor.grad_fn != nothing
                power_tensor.grad_fn(power_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [2.0, 4.0, 6.0]
        end
    
        # Test multiplication by scalar (*, scalar first)
        @testset "Multiplication Function (Scalar * Tensor)" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            mult_tensor = 2.0 * array_tensor
            @test mult_tensor.data == [2.0, 4.0, 6.0]
            @test mult_tensor.requires_grad == true
            @test mult_tensor.shape == (3,)
            @test mult_tensor.op == "*"
    
            # Set gradient of resulting tensor and manually backpropagate
            mult_tensor.grad = [1.0, 1.0, 1.0]
            if mult_tensor.grad_fn != nothing
                mult_tensor.grad_fn(mult_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [2.0, 2.0, 2.0]
        end
    
        # Test multiplication by scalar (*, tensor first)
        @testset "Multiplication Function (Tensor * Scalar)" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            mult_tensor = array_tensor * 3.0
            @test mult_tensor.data == [3.0, 6.0, 9.0]
            @test mult_tensor.requires_grad == true
            @test mult_tensor.shape == (3,)
            @test mult_tensor.op == "*"
    
            # Set gradient of resulting tensor and manually backpropagate
            mult_tensor.grad = [1.0, 1.0, 1.0]
            if mult_tensor.grad_fn != nothing
                mult_tensor.grad_fn(mult_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [3.0, 3.0, 3.0]
        end
    
        # Test division by scalar (/)
        @testset "Division Function (Tensor / Scalar)" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            div_tensor = array_tensor / 2.0
            @test div_tensor.data == [0.5, 1.0, 1.5]
            @test div_tensor.requires_grad == true
            @test div_tensor.shape == (3,)
            @test div_tensor.op == "*"
    
            # Set gradient of resulting tensor and manually backpropagate
            div_tensor.grad = [1.0, 1.0, 1.0]
            if div_tensor.grad_fn !== nothing
                div_tensor.grad_fn(div_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [0.5, 0.5, 0.5]
        end
    
        # Test addition with scalar (+)
        @testset "Addition Function" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            add_tensor = array_tensor + 5.0
            @test add_tensor.data == [6.0, 7.0, 8.0]
            @test add_tensor.requires_grad == true
            @test add_tensor.shape == (3,)
            @test add_tensor.op == "+"
    
            # Set gradient of resulting tensor and manually backpropagate
            add_tensor.grad = [1.0, 1.0, 1.0]
            if add_tensor.grad_fn !== nothing
                add_tensor.grad_fn(add_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [1.0, 1.0, 1.0]
        end
    
        # Test subtraction with scalar (-)
        @testset "Subtraction Function" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            sub_tensor = array_tensor - 1.0
            @test sub_tensor.data == [0.0, 1.0, 2.0]
            @test sub_tensor.requires_grad == true
            @test sub_tensor.shape == (3,)
            @test sub_tensor.op == "+"
    
            # Set gradient of resulting tensor and manually backpropagate
            sub_tensor.grad = [1.0, 1.0, 1.0]
            if sub_tensor.grad_fn !== nothing
                sub_tensor.grad_fn(sub_tensor)
            end
            # Check that gradients in the original tensor are updated correctly
            @test array_tensor.grad == [1.0, 1.0, 1.0]
        end
    
        # Test isless comparison with scalar
        @testset "isless Function" begin
            MiniML.zerograd!(array_tensor)  # Wipe previous grads
            isless_tensor = isless(array_tensor, 2.5)
            @test isless_tensor.data == [1, 1, 0]
            @test isless_tensor.requires_grad == true
            @test isless_tensor.shape == (3,)
            @test isless_tensor.op == "<"
        end
    end    

    @testset "Activation Functions" begin
        
        @testset "Sigmoid" begin
            # Correctness Test
            input = tensor([-100.0, 0.0, 100.0])
            output = MiniML.sigmoid(input)
            @test output.data ≈ [0.0, 0.5, 1.0] atol=1e-6
            # numerical stability
            large_input = tensor([1000.0])
            large_output = MiniML.sigmoid(large_input)
            @test large_output.data ≈ [1.0] atol=1e-6
        end
    
        @testset "ReLU" begin
            input = tensor([-3.0, 0.0, 3.0])
            output = MiniML.relu(input)
            @test isequal(output, tensor([0.0, 0.0, 3.0]))

            large_input = tensor([1e10, -1e10])
            large_output = MiniML.relu(large_input)
            @test isequal(large_output, tensor([1e10, 0.0]))
            
            large_input = tensor([1e10, -1e10])
            large_output = MiniML.relu(large_input)
            @test isequal(large_output, tensor([1e10, 0.0]))
        end
        
        @testset "Softmax Function" begin
            input = tensor([1.0 2.0 3.0; 1.0 2.0 3.0])
            output = MiniML.softmax(input, dim=2)
            @test all(sum(output.data, dims=2) .≈ 1.0)
            
            large_input = tensor([1e10; 1e10 + 1; 1e10 + 2])
            stable_output = MiniML.softmax(large_input, dim=2)
            @test all(sum(stable_output.data, dims=2) .≈ 1.0)
            
            small_input = tensor([-1e10; -1e10 - 1; -1e10 - 2])
            stable_output = MiniML.softmax(small_input, dim=2)
            @test all(sum(stable_output.data, dims=2) .≈ 1.0)
        end
    end
    

end
