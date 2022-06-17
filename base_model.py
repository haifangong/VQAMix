"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import torch
import torch.nn as nn
from attention import BiAttention, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from utils import tfidf_loading
from simple_cnn import SimpleCNN
from learner import MAML
from auto_encoder import Auto_Encoder_Model

# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb, ae_v_emb):
        super(BAN_Model, self).__init__()
        self.args = args
        self.dataset = dataset
        self.op = args.op
        self.glimpse = args.gamma
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:  # if do not use counter
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()
        if args.maml:
            self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            if 'RAD' in args.RAD_dir:
                self.convert = nn.Linear(16384, 64)
            else:
                self.convert = nn.Linear(16384, 32)
            # self.convert = nn.Linear(7056, 64) # use v[0] for ae
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward(self, v, q, lam=None, index=None, test=False, organ=None):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        self.gradients = []

        # get visual feature
        if self.args.maml:
            maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
        if self.args.autoencoder:
            encoder = self.ae_v_emb.forward_pass(v[1])
            ae_feat = encoder
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)

        # get lextual feature
        w_emb = self.w_emb(q)
        if not self.args.use_ablation_q:
            if lam != None:
                w_emb = lam * w_emb + (1-lam) * w_emb[index, :]

        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        q_feat = q_emb

        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        if self.args.use_grad_cam:
            return  q_emb.sum(1), decoder, maml_v_emb, ae_feat, q_feat
        if self.args.autoencoder:
            return q_emb.sum(1), decoder
        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)


# Create SAN model
class SAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb):
        super(SAN_Model, self).__init__()
        self.args = args
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        if args.maml:
            self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
        if 'RAD' in args.RAD_dir:
            self.convert = nn.Linear(16384, 64)
        else:
            self.convert = nn.Linear(16384, 32)
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward(self, v, q, lam=None, index=None, test=False, organ=None):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        self.gradients = []

        # get visual feature
        if self.args.maml:
            maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
        if self.args.autoencoder:
            ae_feat = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(ae_feat)
            ae_v_emb = ae_feat.view(ae_feat.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        w_emb = self.w_emb(q)
        if lam != None:
            w_emb = lam * w_emb + (1-lam) * w_emb[index, :]
        q_emb = self.q_emb(w_emb)  # [batch, q_dim], return final hidden state
        q_feat = q_emb
        att = self.v_att(v_emb, q_emb)
        if self.args.use_grad_cam:
            return  att, decoder, maml_v_emb, ae_feat, q_feat
        if self.args.autoencoder:
            return att, decoder
        return att

    def classify(self, input_feats):
        return self.classifier(input_feats)


# Build BAN model
def build_BAN(dataset, args, priotize_using_counter=False):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0,  args.rnn)
    v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    # build and load pre-trained MAML model
    if args.maml:
        weight_path = args.RAD_dir + '/' + args.maml_model_path
        print('load initial weights MAML from: %s' % (weight_path))
        if 'RAD' in args.RAD_dir:
            maml_v_emb = SimpleCNN(weight_path[:-4]+'.weights', args.eps_cnn, args.momentum_cnn)
        else:
            maml_v_emb = MAML(args.RAD_dir)
            maml_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.RAD_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s'%(weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # Optional module: counter for BAN
    use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter
    if use_counter or priotize_using_counter:
        objects = 10  # minimum number of boxes
    if use_counter or priotize_using_counter:
        counter = Counter(objects)
    else:
        counter = None
    
    # init BAN residual network
    b_net = []
    q_prj = []
    c_prj = []
    for i in range(args.gamma):
        b_net.append(BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1))
        q_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        if use_counter or priotize_using_counter:
            c_prj.append(FCNet([objects + 1, args.num_hid], 'ReLU', .0))
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         ae_v_emb)
    elif args.maml:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         None)
    elif args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None,
                         ae_v_emb)
    return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None, None)


# Build SAN model
def build_SAN(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0, args.rnn)
    v_att = StackedAttention(args.num_stacks, dataset.v_dim, args.num_hid, args.num_hid, dataset.num_ans_candidates,
                             args.dropout)
    # build and load pre-trained MAML model
    if args.maml:
        weight_path = args.RAD_dir + '/' + args.maml_model_path
        print('load initial weights MAML from: %s' % (weight_path))
        if 'RAD' in args.RAD_dir:
            maml_v_emb = SimpleCNN(weight_path[:-4]+'.weights', args.eps_cnn, args.momekntum_cnn)
        else:
            maml_v_emb = MAML(args.RAD_dir)
            maml_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.RAD_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s'%(weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, dataset.num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb)
    elif args.maml:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, None)
    elif args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, ae_v_emb)
    return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, None)