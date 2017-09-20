import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralFingerprint(nn.Module):
	def __init__(self):
		super(NeuralFingerprint, self).__init__()

	def __init__(self, paras):
		self.__init__()
		self.paras = paras
		self.embedding = nn.Embedding(paras.num_feat, paras.F)
		self.hidden = {i: nn.Linear(paras.F, paras.F) for i in range(paras.R)}
		self.output_fp = {i: nn.Linear(paras.F, paras.L) for i in range(paras.R)}
		self.output_cl = nn.Linear(paras.L, paras.num_label)

	def forward(self, graph):
		f = torch.FloatTensor(self.paras.L).zero_()
		embedding = F.torch.unsqueeze(self.embedding(graph.node_feature), 0)
		for L in range(self.paras.R):
			tmp_embedding = []
			# graph.nodes must be a sorted list
			for node in graph.nodes:
				v = embedding[node]
				for neighbour in graph.neighbours(v):
					v += embedding[neighbour]
				r = F.relu(self.hidden[L](v))
				tmp_embedding.append(r)
				i = F.softmax(self.output_fp[L](r))
				f += i
			for i, r in enumerate(tmp_embedding):
				embedding[i] = r
		return F.softmax(self.output_fp(f))
