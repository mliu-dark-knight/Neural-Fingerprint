from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from util import xavier_uniform

class NeuralFingerprint(nn.Module):
	def __init__(self, paras):
		super(NeuralFingerprint, self).__init__()
		self.paras = paras
		self.embedding = nn.Embedding(paras.num_feat, paras.F)
		self.hidden = {i: nn.Linear(paras.F, paras.F) for i in range(paras.R)}
		self.output_fp = {i: nn.Linear(paras.F, paras.L) for i in range(paras.R)}
		self.output_cl = nn.Linear(paras.L, paras.num_class)

	def weight_init(self):
		xavier_uniform(self.embedding)
		for _, hidden in self.hidden:
			init.xavier_normal(hidden)
		for _, output in self.output_fp:
			init.xavier_normal(output)
		init.xavier_normal(self.output_cl)

	def forward(self, graph):
		f = Variable(torch.FloatTensor(self.paras.L).zero_(), requires_grad=True)
		embedding = self.embedding(Variable(torch.from_numpy(graph.node_feature)))
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
			embedding = torch.stack(tmp_embedding, 0)
		return F.log_softmax(self.output_cl(f))
