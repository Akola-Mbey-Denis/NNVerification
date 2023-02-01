#############################################################
# Denis Mbey Akola
# CPS Master 2
############################################################
using NeuralVerification, LazySets
using Symbolics 
using StaticArrays
using IntervalArithmetic
using LinearAlgebra
import NeuralVerification: Network, Layer, ReLU, Sigmoid, Id, compute_output, ActivationFunction, get_bounds,affine_map
include(ARGS[1])

print("Computing approximations for R={z | ")
i = 1
for q in quantifiers
    print(q)
    print(" x[")
    print(i)
    print("] ")
    global i = i+1
end
print("z = ")
print(g)
println("}")

g_expr = build_function(g, [x[i] for i=1:p])
my_g = eval(g_expr)

# input hyperrectangle, coded as a static array
input = @SVector [@interval(-1.0,1.0) for i = 1:p]    
input_center = zeros(p)

c = my_g(input_center)

Dg = Symbolics.jacobian([g], [x[i] for i=1:p])
Dg_expr = build_function(Dg, [x[i] for i=1:p])
my_Dg = eval(Dg_expr[1])

range_Dg = my_Dg(input)

function O(range_Jf, i)
    # convenience function returning the outer-approximated interval for the contribution of xi, i between 1 and dim input, to function f for which we give the range of the Jacobian range_Jf
    # Take the upper bound of the absolute value of the interval
    max_ = abs(range_Jf[i]).hi
    return @interval(-max_ ,max_)    
    end

function I(range_Jf, i)
    # convenience function returning the inner-approximated interval for the contribution of xi, i between 1 and dim input, to function f for which we give the Jacobian Jf
     
    # Take the lower bound of the absolute value of the interval
    min_ = abs(range_Jf[i]).lo
    return @interval(-min_ ,min_)
end

function sigmoid(x::Float64)
   return 1.0/(1.0+exp(-x))
end

function sigmoid_act(x::Hyperrectangle)
    # Since sigmoid is monotonic, we apply it  bound wise
    u_x = LazySets.high(x)
    l_x = LazySets.low(x) 
    lower = zeros(Float64,length(l_x))
    high =  zeros(Float64,length(u_x))

    for j in 1: length(u_x)
        lower[j] = sigmoid(l_x[j])
        high[j]  = sigmoid(u_x[j])
    end
   return Hyperrectangle(low=lower, high=high)
end


function sigmoidder(x::Float64)

    # Deriving the derivative of sigmoid yield (1-sigmoid(x))*sigmoid(x)
    
    return (1.0 - sigmoid(x))*sigmoid(x)
 end

# Hack for considering also nnet networks to be sigmoid activated
function act_gradient(act::ReLU, R::Hyperrectangle)
    
    # Retrieve the upper and lower bounds
    lower_bound = LazySets.low(R)
    upper_bound = LazySets.high(R)
   
    lower = zeros(Float64,length(lower_bound))
    high =  zeros(Float64,length(upper_bound))

    for j in 1: length(lower_bound)
        #verify if  upper bound is greater than 0
        if upper_bound[j]<0.0 
            lower[j] = sigmoidder(lower_bound[j])
            high[j] = sigmoidder(upper_bound[j])

        # verify if lower bound is greater than 0
        elseif lower_bound[j]>0.0
            lower[j] = sigmoidder(upper_bound[j])
            high[j] = sigmoidder(lower_bound[j])
        # case for sigmoid result being 0
        else
            high[j] = 0.25 
            lower[j] = min(sigmoidder(lower_bound[j]),sigmoidder(upper_bound[j]))
        end
    end
  return Hyperrectangle(low=lower, high=high)
  end
  
  act_gradient(act::Sigmoid, R::Hyperrectangle) = act_gradient(ReLU, R)
  act_gradient(act::Id, R::Hyperrectangle) = R



  """
  get_gradient(nnet::Network, x::Vector)

Given a network, find the gradient at the input x
"""
function get_gradient(nnet::Network, x::Vector)
  z = x
  gradient = fill(@interval(0.0,0.0),length(x), length(x))
  for (i, layer) in enumerate(nnet.layers)
    z_hat = affine_map(layer, z)
    m_gradient = act_gradient(layer.activation, z_hat)
    gradient = Diagonal(m_gradient) * layer.weights * gradient
    z = layer.activation(z_hat)
  end
  return gradient
  end


  function Diagonal(x::Hyperrectangle)
    # Diagonal matrix of intervals
    lower = LazySets.low(x)
    high  = LazySets.high(x)
    mat = fill(@interval(0.0,0.0),length(lower),length(lower))
    for i in 1 : length(lower)
        mat[i,i]=@interval(lower[i],high[i])  
    end
  return mat
  end


function  interval_gradient(nnet::Network, x::Hyperrectangle)
    z = x
    gradient = fill(@interval(0.0,0.0),length(LazySets.high(x)), length(LazySets.high(x)))   
    for (i, layer) in enumerate(nnet.layers)
        z_hat = LazySets.affine_map(layer.weights, z, layer.bias)
        # Get an over approximation of hyperrectangle
        box_approx = box_approximation(z_hat)

        # Gradient computation
        m_gradient = act_gradient(layer.activation, box_approx)
        gradient = Diagonal(m_gradient) * layer.weights * gradient
        
        # Apply appropriate activation function
        if layer.activation == Id()
            z = layer.activation(box_approx)
        else
            z = sigmoid_act(box_approx)
        end
    return gradient

    end 

end


net = read_nnet("model.nnet")
X = Hyperrectangle(low = [-1.0,-1.0], high = [1.0, 1.0]) 
print(interval_gradient(net,X))