import torch

checkpoint = torch.load(r'out\XRMB\SupervisedDataFusionopt_acc.pt')
print(checkpoint['epoch'])
print(checkpoint['train_loss'])
print(checkpoint['valid_acc'])