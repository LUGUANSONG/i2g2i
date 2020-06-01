# import torch
# import torch.nn as nn
#
#
# class net(nn.Module):
#     def __init__(self, dimension):
#         super(net, self).__init__()
#
#         self.w = nn.Parameter(torch.ones(dimension))
#
#     def forward(self, x, y=None):
#         x = x * self.w
#         return x ** 2, x
#
#
# x_hat = torch.Tensor([0,1,2,3])
# x_hat.requires_grad_(True)
# f = net(x_hat.size(0))
#
# x_hat_score = f(x_hat)
# if len(x_hat_score) == 2:
#     x_hat_score = x_hat_score[0]
# if x_hat_score.dim() > 1:
#     x_hat_score = x_hat_score.view(x_hat_score.size(0), -1).mean(dim=1)
# gradients = torch.autograd.grad(outputs=x_hat_score, inputs=x_hat,
#                                 grad_outputs=torch.ones(x_hat_score.size()),
#                                 create_graph=True, retain_graph=True, only_inputs=True)
# print(gradients)
#
# gradients = gradients[0].view(x_hat.size(0), -1)  # flat the data
#
# print(gradients)


from argparse import ArgumentParser

parser = ArgumentParser(description='training code')
# Options to deprecate
parser.add_argument('--l1', type=float)
args = parser.parse_args()

print(getattr(args, 'l1'))

setattr(args, 'l1', 9)
print(getattr(args, 'l1'))
