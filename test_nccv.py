import torch
print(torch.version.cuda)
a=torch.cuda.nccl.version()
print(a)