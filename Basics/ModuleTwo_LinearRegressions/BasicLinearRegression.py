import torch
from torch.nn import Linear
torch.manual_seed(1)
# The require_grad option here indicates that these are values that must be learned.
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad=True)

def forward(x):
    y = w * x + b
    return y

x = torch.tensor([1.0])

# Prediction variable
y_hat = forward(x)

print("Y_hat: ", y_hat)

x = torch.tensor([[1],[2]])
y_hat = forward(x)

print("Y_hat: ", y_hat)

# Use the torch.nn Linear library to define a model.
# The in_features and out_features paramters specify the size of the inputs and outputs.
model = Linear(in_features = 1, out_features = 1)

# See what the model parameters are initialized to.
# The first parameter is the slope. The second is the bias.
print("\nModel parameters: ", list(model.parameters()))

x = torch.tensor([0.0])
y_hat = model(x) # Note that a call to model() automatically applies a 'forward' transform.
print("When input is ", x, " \n\tprediction is: ", y_hat)

x = torch.tensor([[1.0],[2.0]])
y_hat = model(x)
print("\nWhen input is ", x, " \n\tprediction is: ", y_hat)
