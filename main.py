import argparse
from predictor import Predictor

def parse_args():
	parser = argparse.ArgumentParser(description="Neural Fingerprint")
	parser.add_argument('--dataset', default='enzymes', help=None)
	parser.add_argument('--F', default=64, help='Embedding dimension')
	parser.add_argument('--R', default=4, help='Number of hidden layers')
	parser.add_argument('--L', default=64, help='Fingerprint length')
	parser.add_argument('--num_feat', default=-1, help='Number of features')
	parser.add_argument('--num_class', default=-1, help='Number of class labels')
	parser.add_argument('--learning_rate', default=1e-3, help=None)
	parser.add_argument('--epoch', default=1, help=None)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	predictor = Predictor(args)
	predictor.train()
	predictor.test()
