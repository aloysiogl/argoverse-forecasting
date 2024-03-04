import tensorflow as tf
import torch 
# print(tf.test.is_gpu_available(
#     cuda_only=False, min_cuda_compute_capability=None
# ))
# print("========================")
# print (tf.test.gpu_device_name())
print(torch.cuda.is_available())
print(torch.version.cuda)
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
print(t,r,a,f)
print(torch.cuda.mem_get_info())
print(torch.cuda.memory_summary(device=None, abbreviated=False))