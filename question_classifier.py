import torch
import torch.nn as nn
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier

class Question_Classifier(nn.Module):
	"""docstring for Question_Classifier"""
	def __init__(self, args, w_emb, q_emb, classifier):
		super(Question_Classifier, self).__init__()
		self.args = args
		self.w_emb = w_emb
		self.q_emb = q_emb
		self.classifier = classifier

	def forward(self, q, lam=None, index=None):
		w_emb = self.w_emb(q)
		if lam != None:
			w_emb = lam * w_emb + (1-lam) * w_emb[index, :]
		q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
		return self.classifier(q_emb.sum(1))

def build_QC(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0,  args.rnn)
    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)
    return Question_Classifier(args, w_emb, q_emb, classifier)