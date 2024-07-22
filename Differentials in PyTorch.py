# Basic examples of calculating derivatives in PyTorch

import torch

# The requires_grad parameter specifies that this variable will be used for
# evaluating functions and derivatives
x = torch.tensor(2.0, requires_grad=True)
y = x**2


print("X: \n",x)
print("Y: \n",y)

# The backward function calculates the derivative of y and evaluates it for the value of X.
# In the case, the derivative of Y is 2X^1
y.backward()

print("The derivative of Y: \n", y)
print("The gradient of X: \n", x.grad)

# Evaluating a function can be done with respect to different variables in that function
# (In other words... brush up on your calculus, or rewatch these slides:)
# https://www.coursera.org/learn/deep-neural-networks-with-pytorch/lecture/2Mwmu/differentiation-in-pytorch
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u*v + u**2

print("U:\n", u)
print("V:\n", v)
print("F:\n", f)
# Calling the backward() function on F will calculate the two partial derivatives of F
# Note that the result shows two values (the derivatives with respect to U and V).
f.backward()
print("The derivatives of F:\n", f)

# Calling the grad function on U will calculate the derivative of F with respect to U
# (that's what the slides say... but will it?)
print("Derivative of F with respect to U:\n", u.grad)