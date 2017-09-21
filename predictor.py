from __future__ import print_function
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import NeuralFingerprint
from torch.autograd import Variable

class Graph(object):
	def __init__(self, graph, label, feature_map):
		self.label = np.asarray([label], dtype=int)
		self.nodes = np.asarray([-1] * len(graph), dtype=int)
		self.node_feature = np.asarray([-1] * len(graph), dtype=int)
		self.neighbours = np.array([None] * len(graph))
		self.init(graph, feature_map)

	def init(self, graph, feature_map):
		for node, info in graph.items():
			self.nodes[node] = node
			self.neighbours[node] = np.asarray(info['neighbors'], dtype=int)
			self.node_feature[node] = feature_map[info['label']]


class Predictor(object):
	def __init__(self, paras):
		self.paras = paras
		self.data = self.load_data()
		self.model = NeuralFingerprint(paras)

	def map_feat_class(self):
		feature_map = {}
		class_map = {}
		file = 'classification-datasets/' + self.paras.dataset + '/' + self.paras.dataset + '.graph'
		with open(file) as f:
			graphs = pickle.loads(f.read())
		num_feature = 0
		for _, graph in graphs['graph'].items():
			for _, info in graph.items():
				label = info['label']
				assert len(label) == 1
				if label not in feature_map:
					feature_map[label] = num_feature
					num_feature += 1
		num_class = 0
		for label in graphs['labels']:
			if label not in class_map:
				class_map[label] = num_class
				num_class += 1
		return feature_map, class_map

	def load_data(self):
		data = []
		file = 'classification-datasets/' + self.paras.dataset + '/' + self.paras.dataset + '.graph'
		feature_map, class_map = self.map_feat_class()
		self.paras.num_feat, self.paras.num_class = len(feature_map), len(class_map)
		with open(file) as f:
			graphs = pickle.loads(f.read())
		for id, graph in graphs['graph'].items():
			data.append(Graph(graph, class_map[graphs['labels'][id]], feature_map))
		return data

	def train(self):
		optimizer = optim.Adam(self.model.parameters(), lr=self.paras.learning_rate)
		self.model.train()
		for _ in tqdm(range(self.paras.epoch), ncols=100):
			for i in tqdm(range(len(self.data)), ncols=100):
				graph = self.data[i]
				optimizer.zero_grad()
				output = self.model(graph).unsqueeze(0)
				target = Variable(torch.from_numpy(graph.label))
				loss = F.nll_loss(output, target)
				loss.backward()
				optimizer.step()

	def test(self):
		correct = 0.0
		for i in tqdm(range(len(self.data)), ncols=100):
			graph = self.data[i]
			target = torch.from_numpy(graph.label)
			pred = self.model(graph).data.max(dim=0)[1]
			correct += pred.eq(target.view_as(pred)).cpu().sum()
		print('Accuracy: %f' % (correct / len(self.data)))
