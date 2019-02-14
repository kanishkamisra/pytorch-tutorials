import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# tensors = generalizations of Matrices.
# 1d tensor (vector)
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)

# 2d Tensor (matrix)
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.Tensor(M_data) 
print(M)

# 3d Tensors
T_data = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)
print(T)

x = torch.randn(3, 4, 5)
print(x)

# Tensor Operations
x = torch.Tensor([1, 2, 3])
y = torch.Tensor([4, 5, 6])

z = x + y
print(z)

# concatenate tensors
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# along columns
x_2 = torch.randn(2, 5)
y_2 = torch.randn(2, 4)
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# reshaping tensors
x = torch.randn(2, 3, 4)
print(x)

print(x.view(2, 12))
# provide all but 1 dimension, and torch infers what the other dimension is :)
print(x.view(2, -1))

## Computation Graphs and Auto-diff
print("\nComputation Graphs and Auto Differentiation\n")
x = autograd.Variable(torch.Tensor([1, 2, 3]), requires_grad = True)
print(x.data)
y = autograd.Variable(torch.Tensor([4, 5, 6]), requires_grad = True)
z = x + y

print(z.data)
print(z.grad_fn)
