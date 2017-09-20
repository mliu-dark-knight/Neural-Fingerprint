import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NeuralFingerprint(nn.Module):
	def __init__(self, paras):
		super(NeuralFingerprint, self).__init__()
		self.paras = paras
		self.embedding = nn.Embedding(paras.num_feat, paras.F)
		self.hidden = {i: nn.Linear(paras.F, paras.F) for i in range(paras.R)}
		self.output_fp = {i: nn.Linear(paras.F, paras.L) for i in range(paras.R)}
		self.output_cl = nn.Linear(paras.L, paras.num_class)

	def forward(self, graph):
		f = Variable(torch.FloatTensor(self.paras.L).zero_(), requires_grad=True)
		embedding = F.torch.unsqueeze(self.embedding(Variable(torch.from_numpy(graph.node_feature))), 1)
		for L in range(self.paras.R):
			tmp_embedding = []
			# graph.nodes must be a sorted list
			for node in graph.nodes:
				v = embedding[node]
				for neighbour in graph.neighbours[node]:
					v = torch.add(v, embedding[neighbour])
				r = F.relu(self.hidden[L](v))
				tmp_embedding.append(r)
				i = F.softmax(self.output_fp[L](r))
				f = torch.add(f, i)
			for i, r in enumerate(tmp_embedding):
				embedding[i] = r
		return F.softmax(self.output_cl(f))
