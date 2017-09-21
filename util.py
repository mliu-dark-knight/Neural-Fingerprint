import math
import torch.nn.init as init

def xavier_uniform(tensor, gain=1):
	fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
	std = gain * math.sqrt(1.0 / fan_out)
	a = math.sqrt(3.0) * std
	return tensor.uniform_(-a, a)
