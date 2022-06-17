"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import argparse
import torch
from torch.utils.data import DataLoader
import dataset_RAD
import dataset_VQA
import base_model
import utils
import pandas as pd
import os
import json
import numpy as np
from PIL import Image
import cv2
from ece import _ECELoss, _MCELoss, _OELoss
from torchvision.transforms.functional import to_pil_image

answer_types = ['CLOSED', 'OPEN', 'ALL']
quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/SAN_MEVF',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=int, default=79,
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    # Choices of Attention models
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

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
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

    parser.add_argument('--use_ablation', action='store_true', default=False)
    parser.add_argument('--use_ablation_v', action='store_true', default=False)
    parser.add_argument('--use_ablation_q', action='store_true', default=False)

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=True,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.pth',
                        help='the maml_model_path we use')
    # parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
    #                     help='the maml_model_path we use')

    parser.add_argument('--use_grad_cam', action='store_true', default=False)

    # Return args
    args = parser.parse_args()
    return args


# Load questions
def get_question(q, dataloader):
    q = q.squeeze(0)
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


# Load answers
def get_answer(p, dataloader):
    _m, idx = p.max(1)
    return dataloader.dataset.label2ans[idx.item()]


def get_result_pathVQA(model, dataloader, device, args):
    ece_criterion = _ECELoss()
    mce_criterion = _MCELoss()
    oe_criterion = _OELoss()
    logits_list = []
    labels_list = []
    all_batch_score = 0
    ans_types = ['other', 'yes/no', 'all']
    keys = ['correct', 'total', 'score']
    result = dict((i, dict((j, 0.0) for j in keys)) for i in ans_types)
    answers_list = []
    model.eval()
    with torch.no_grad():
        # if True:
        import time
        t = time.time()
        total_time = 0.0
        for v, q, a, ans_type, q_type, p_type, _, _, _ in iter(dataloader):
            if args.maml:
                v[0] = v[0].reshape(1, 3, 84, 84)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            q = q.to(device)
            a = a.to(device)
            # inference and get logit
            if args.autoencoder:
                features, _ = model(v, q)
            else:
                features = model(v, q)
            preds = model.classifier(features)

            final_preds = preds

            logits_list.append(final_preds)
            _, lbl = torch.max(a, 1)
            labels_list.append(lbl)

            batch_score = compute_score_with_logits(final_preds, a.data).sum()
            all_batch_score += batch_score.item()

            answer = {}
            answer['answer_type'] = ans_type[0]
            answer['predict'] = get_answer(final_preds, dataloader)
            answer['ref'] = get_answer(a, dataloader)
            answers_list.append(answer)
            # Compute accuracy for each type answer
            if ans_type[0] == "yes/no":
                result[ans_type[0]]['correct'] += float(batch_score)
                result[ans_type[0]]['total'] += 1
            else:
                result['other']['correct'] += float(batch_score)
                result['other']['total'] += 1
            result['all']['correct'] += float(batch_score)
            result['all']['total'] += 1
            total_time += time.time() - t
            t = time.time()
        print('time <s/sample>: ', total_time / result['all']['total'])
    result['yes/no']['score'] = result['yes/no']['correct'] / result['yes/no']['total']
    result['other']['score'] = result['other']['correct'] / result['other']['total']
    result['all']['score'] = result['all']['correct'] / result['all']['total']

    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()
    print('Overall ACC:', all_batch_score * 100 / len(dataloader))
    ece = ece_criterion(logits, labels).item()
    print('ECE: {}'.format(ece))
    mce = mce_criterion(logits, labels).item()
    print('MCE: {}'.format(mce))
    oe = oe_criterion(logits, labels).item()
    print('OE: {}'.format(oe))
    return result, answers_list


# Logit computation (for train, test or evaluate)
def get_result(model, dataloader, device, args):
    ece_criterion = _ECELoss()
    mce_criterion = _MCELoss()
    oe_criterion = _OELoss()
    logits_list = []
    labels_list = []
    err_score = {'CLOSED': [], 'OPEN': []}

    keys = ['count', 'real', 'true', 'real_percent', 'score', 'score_percent']
    question_types_result = dict(
        (i, dict((j, dict((k, 0.0) for k in keys)) for j in quesntion_types)) for i in answer_types)
    result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)
    model.eval()
    if args.use_grad_cam:
        model.q_emb.train()

    tot = 0
    num = 0
    all_batch_score = 0
    import time
    with torch.no_grad():
        for v, q, a, ans_type, q_types, p_type, qid, img_name, _ in iter(dataloader):
            if p_type[0] != "freeform":
                continue
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            q = q.to(device)
            a = a.to(device)
            t = time.time()
            # inference and get logit
            if args.use_grad_cam:
                features, _, maml_feat, ae_feat, q_feat = model(v, q)
            elif args.autoencoder:
                features, _ = model(v, q, test=True)
            else:
                features = model(v, q, test=True)
            preds = model.classifier(features)
            sub = time.time() - t
            tot += sub
            num += 1
            final_preds = preds
            logits_list.append(final_preds)
            _, lbl = torch.max(a, 1)
            labels_list.append(lbl)
            batch_score = compute_score_with_logits(final_preds, a.data).sum()
            all_batch_score += batch_score.item()
            if batch_score == 0:
                scores = torch.nn.functional.softmax(final_preds, dim=1).detach().cpu().numpy()
                err_score[ans_type[0]].append(float(scores.max()))
            # Compute accuracy for each type answer
            result[ans_type[0]]['count'] += 1.0
            result[ans_type[0]]['true'] += float(batch_score)
            result[ans_type[0]]['real'] += float(a.sum())

            result['ALL']['count'] += 1.0
            result['ALL']['true'] += float(batch_score)
            result['ALL']['real'] += float(a.sum())

            q_types = q_types[0].split(", ")
            for i in q_types:
                question_types_result[ans_type[0]][i]['count'] += 1.0
                question_types_result[ans_type[0]][i]['true'] += float(batch_score)
                question_types_result[ans_type[0]][i]['real'] += float(a.sum())

                question_types_result['ALL'][i]['count'] += 1.0
                question_types_result['ALL'][i]['true'] += float(batch_score)
                question_types_result['ALL'][i]['real'] += float(a.sum())

    mean_time = tot / num
    print('mean_time', mean_time)
    print('Overall ACC:', all_batch_score / 3.08)

    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()
    ece = ece_criterion(logits, labels).item()
    print('ECE: {}'.format(ece))
    mce = mce_criterion(logits, labels).item()
    print('MCE: {}'.format(mce))
    oe = oe_criterion(logits, labels).item()
    print('OE: {}'.format(oe))

    for i in answer_types:
        result[i]['score'] = result[i]['true'] / result[i]['count']
        result[i]['score_percent'] = round(result[i]['score'] * 100, 1)
        for j in quesntion_types:
            if question_types_result[i][j]['count'] != 0.0:
                question_types_result[i][j]['score'] = question_types_result[i][j]['true'] / \
                                                       question_types_result[i][j]['count']
                question_types_result[i][j]['score_percent'] = round(question_types_result[i][j]['score'] * 100, 1)
            if question_types_result[i][j]['real'] != 0.0:
                question_types_result[i][j]['real_percent'] = round(
                    question_types_result[i][j]['real'] / question_types_result[i][j]['count'] * 100.0, 1)
    return result, question_types_result, err_score


# Test phase
if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    if 'RAD' in args.RAD_dir:
        args.feat_dim = 64
    else:
        args.feat_dim = 32
    # Check if evaluating on TDIUC dataset or VQA dataset
    if 'RAD' not in args.RAD_dir:
        dictionary = dataset_VQA.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
        eval_dset = dataset_VQA.VQAFeatureDataset('test', args, dictionary)
    else:
        dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
        eval_dset = dataset_RAD.VQAFeatureDataset(args.split, args, dictionary)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args)
    eval_loader = DataLoader(eval_dset, 1, shuffle=False, num_workers=0, pin_memory=False,
                             collate_fn=utils.trim_collate)


    def save_questiontype_results(outfile_path, quesntion_types_result):
        for i in quesntion_types_result:
            pd.DataFrame(quesntion_types_result[i]).transpose().to_csv(outfile_path + '/question_type_' + i + '.csv')

    # Testing process
    def process(args, model, eval_loader):
        if os.path.exists(args.input + '/model_epoch_best.pth'):
            model_path = args.input + '/model_epoch_best.pth'
        else:
            model_path = args.input + '/model_epoch%s.pth' % args.epoch
        print('loading %s' % model_path)
        model_data = torch.load(model_path)
        print('dataloader', len(eval_loader))

        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        if args.use_RAD:
            if 'RAD' in args.RAD_dir:
                result, quesntion_types_result, err_score = get_result(model, eval_loader, args.device, args)
            else:
                result, quesntion_types_result = get_result_pathVQA(model, eval_loader, args.device, args)

            outfile_path = args.output + '/' + args.input.split('/')[1]
            outfile = outfile_path + '/results.json'
            if not os.path.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            json.dump(result, open(outfile, 'w'))
        return

    process(args, model, eval_loader)
