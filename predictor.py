import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import NeuralFingerprint
from torch.autograd import Variable

class Predictor(object):
	def __init__(self, paras):
		self.paras = paras
		self.train_data = None
		self.test_data = None
		self.model = NeuralFingerprint()
		pass

	def train(self):
		optimizer = optim.Adam(self.model.parameters(), lr=self.paras.learning_rate)
		self.model.train()
		for epoch in tqdm(range(self.paras.epoch), ncols=100):
			for data, target in self.train_data:
				optimizer.zero_grad()
				data, target = Variable(data), Variable(target)
				output = self.model(data)
				loss = F.cross_entropy(output, target)
				loss.backward()
				optimizer.step()
