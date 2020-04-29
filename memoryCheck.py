import torch

test = torch.cuda.memory_summary(device='cuda:0', abbreviated=False)
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_cached(device='cuda:0')
torch.cuda.reset_max_memory_allocated(device='cuda:0')
print(test)