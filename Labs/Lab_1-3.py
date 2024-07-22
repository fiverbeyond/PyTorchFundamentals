# This is a private implementation of Lab 3 "Derivatives and Graphs in PyTorch"
# from the Coursera course "Deep Neural Networks with Pytorch".
# MOST OF THE CODE HERE IS NOT MY OWN, but it was typed by me (not copied) as I worked through each line in the lab.


import torch
import matplotlib.pylab as plt

# Create a tensor x
x = torch.tensor(2.0, requires_grad=True)
print("Tensor X:\n", x)

# Create a tensor y equal to y = x^2
y = x ** 2
print("The result of y = X^2:", y)

# Take the derivative of y with respect to x,
# Printing out the value at x = 2

y.backward()
print("The derivative at x = 2: ", x.grad)

print("\n::  X  ::")
print('Data:', x.data)
print('grad_fn:', x.grad_fn)
print('grad:', x.grad)
print("is_leaf:", x.is_leaf)
print("requires_grad:", x.requires_grad)

print("\n::  Y  ::")
print('Data:', y.data)
print('grad_fn:', y.grad_fn)
print('grad:', y.grad)
print("is_leaf:", y.is_leaf)
print("requires_grad:", y.requires_grad)


x = torch.tensor(1.0, requires_grad=True)
y = 2 * (x ** 3) + x

y.backward()

print ("Derivative of y: ", x.grad)

# Partial derivatives
# Calculate f(u, v) = v * u + u^2  at u = 1, v = 2
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u * v + u ** 2
print("The result of v * u + u^2: ", f)

f.backward()
print("The partial derivative with respect to u: ", u.grad)

# Calculate the derivative with respect to v:
print("The partial derivative with respect to v: ", v.grad)

# Calculate the derivative with multiple values

x = torch.linspace(-10, 10, 10, requires_grad = True)
y = x ** 2
y_sum = torch.sum(x ** 2)

# Take the derivative with respect to multiple values.
# Plat out the function and its derivative
y_sum .backward()

plt.plot(x.detach().numpy(), y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

# Apply the ideas in this lab to take the derivative of Relu activations

x = torch.linspace(-1, 10, 1000, requires_grad=True)
y  = torch.relu(x)
y_sum = y.sum()
y_sum.backward()
plt.plot(x.detach().numpy(), y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.xlabel('x')
plt.legend()
plt.show()

# Practice calculating the derivative of f = u * v = (u * v) ** 2 at u = 2, v = 1
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(1.0, requires_grad=True)
f = (u * v) + (u * v) ** 2

f.backward()
print("Derivative of F with respect to U: ", u.grad)



