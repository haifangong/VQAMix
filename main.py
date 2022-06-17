"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa)
"""
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import base_model
import dataset_RAD
import dataset_VQA
import utils
from question_classifier import build_QC
from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models/san_mevf',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=20,
                        help='the number of epoches')
    parser.add_argument('--lr', default=0.02, type=float, metavar='lr',
                        help='initial learning rate')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')

    # Choices of attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - support testing, gpu training or sampling
    parser.add_argument('--print_interval', default=40, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    parser.add_argument('--use_RAD', action='store_true', default=False,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--RAD_dir', type=str,
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=32, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=True,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=True,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.pth',
                        help='the maml_model_path we use')
    # parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
    #                     help='the maml_model_path we use')
    parser.add_argument('--use_mix', action='store_true', default=False)
    parser.add_argument('--use_partial_label', action='store_true', default=False)
    # parser.add_argument('--qc_model_path', type=str, default='./saved_models/QC/path_best.pth')
    parser.add_argument('--qc_model_path', type=str, default='./saved_models/QC/rad_best.pth')
    parser.add_argument('--pow', type=int, default=2) # pow factor

    parser.add_argument('--use_mix_all', action='store_true', default=False)
    parser.add_argument('--use_mix_cond', action='store_true', default=False)
    parser.add_argument('--use_mix_cond_v', action='store_true', default=False)
    parser.add_argument('--use_mix_cond_q', action='store_true', default=False)
    parser.add_argument('--use_mix_cond_vq_in', action='store_true', default=False)
    parser.add_argument('--use_mix_cond_vq_union', action='store_true', default=False)

    parser.add_argument('--use_ablation', action='store_true', default=False)
    parser.add_argument('--use_ablation_v', action='store_true', default=False)
    parser.add_argument('--use_ablation_q', action='store_true', default=False)

    parser.add_argument('--use_rescale', action='store_true', default=False)
    parser.add_argument('--rescale_factor', type=float, default=1.5) # > 1

    parser.add_argument('--use_sep_mix', action='store_true', default=False)
    parser.add_argument('--use_kl_mix', action='store_true', default=False)
    parser.add_argument('--use_grad_cam', action='store_true', default=False) # means using grad-cam-mix in training
    parser.add_argument('--alpha', default=1, type=float)

    parser.add_argument('--five_fold', action='store_true', default=False)

    # Return args
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # create output directory and log file
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())
    # Set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if 'RAD' in args.RAD_dir:
        args.feat_dim = 64
    else:
        args.feat_dim = 32

    # Load dictionary and RAD training dataset
    if 'RAD' not in args.RAD_dir:
        dictionary = dataset_VQA.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
        train_dset = dataset_VQA.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
        eval_dset = dataset_VQA.VQAFeatureDataset('val', args, dictionary, question_len=args.question_len)
    else:
        dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
        train_dset = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
        eval_dset = dataset_RAD.VQAFeatureDataset('test', args, dictionary, question_len=args.question_len)

    batch_size = args.batch_size
    # Create VQA model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args)
    optim = None
    epoch = 0
    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        model.to(device)
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

    # create training dataloader
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=8, collate_fn=utils.trim_collate, pin_memory=True)
    eval_loader = DataLoader(eval_dset, 1, shuffle=False, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)

    qc_model = None
    if args.use_partial_label:
        qc_model = build_QC(train_dset, args)
        qc_model_data = torch.load(args.qc_model_path)
        qc_model.load_state_dict(qc_model_data.get('model_state', qc_model_data))
        print('load {} ...'.format(args.qc_model_path))

    # training phase
    train(args, model, train_loader, eval_loader, args.epochs, args.output, optim, epoch, qc_model)
