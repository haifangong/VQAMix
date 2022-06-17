"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import torch
import contextlib
import numpy as np
import random
import copy
import torch.nn as nn


def mixup_criterion_abla(criterion, pred, y_a, y_b, lam):
    y = lam * y_a + (1 - lam) * y_b
    return criterion(pred, y)


def mixup_criterion(criterion, pred, y_a, y_b, lam, pow=2):
    y = lam ** pow * y_a + (1 - lam) ** pow * y_b
    return criterion(pred, y)


def mixup_criterion_all(criterion, pred, y_a, y_b, lam, pow=2):
    y = lam ** pow * y_a + (1 - lam) ** pow * y_b + lam * (1 - lam) * (y_a + y_b)
    return criterion(pred, y)


def mixup_data(v, a, args, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda without organ constraint'''
    lam = np.random.beta(args.alpha, args.alpha)

    batch_size = v[0].size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_v = [0, 0]
    if args.maml:
        mixed_v[0] = lam * v[0] + (1 - lam) * v[0][index, :]
    if args.autoencoder:
        mixed_v[1] = lam * v[1] + (1 - lam) * v[1][index, :]

    a_1, a_2 = a, a[index]
    return mixed_v, a_1, a_2, lam, index


def obtain_key(i, key_dict):
    for key in key_dict.keys():
        if i in key_dict[key]:
            return key


def modality_specific_mixup_data(v, a, args, cond_vq, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda without organ constraint'''
    random.seed(0)
    np.random.seed(0)
    organ, question = cond_vq
    lam = np.random.beta(args.alpha, args.alpha)

    condition_index_list = []
    v_type_list = {"ABD": [], "HEAD": [], "CHEST": []}
    if 'RAD' in args.RAD_dir:
        q_type_list = {"ABN": [], "PRES": [], "MODALITY": [], "ORGAN": [], "POS": [], "PLANE": [], "COUNT": [],
                       "ATTRIB": [], "COLOR": [], "OTHER": [], "SIZE": []}
    else:
        open_list = ['WHERE', 'WHAT', 'HOW', 'WHEN', 'WHOSE', 'WHO', 'WHY']
        q_type_list = {"OPEN": [], "CLOSED": []}
    if args.use_mix_cond_v:
        for i, j in enumerate(organ):
            v_type_list[j].append(i)
        v_type_list_copy = copy.deepcopy(v_type_list)
        for i in range(len(organ)):
            key = obtain_key(i, v_type_list_copy)
            random_index = random.randint(0, len(v_type_list[key]) - 1)
            value = v_type_list[key].pop(random_index)
            condition_index_list.append(value)

    elif args.use_mix_cond_q:
        for i, j in enumerate(question):
            j = j.split(",")[0].upper()
            if not 'RAD' in args.RAD_dir:
                if j in open_list:
                    j = "OPEN"
                else:
                    j = "CLOSED"
            q_type_list[j].append(i)
        q_type_list_copy = copy.deepcopy(q_type_list)
        for i in range(len(question)):
            key = obtain_key(i, q_type_list_copy)
            random_index = random.randint(0, len(q_type_list[key]) - 1)
            value = q_type_list[key].pop(random_index)
            condition_index_list.append(value)

    elif args.use_mix_cond_vq_in:
        index2q_type = {}
        for index in range(len(question)):
            organ_type = organ[index]
            question_type = question[index].split(",")[0].upper()
            v_type_list[organ_type].append(index)
            index2q_type.update({str(index): question_type})
        intersection_dict = {}
        for key in v_type_list.keys():
            for index in v_type_list[key]:
                q_type = index2q_type[str(index)]
                if key + '_' + q_type not in intersection_dict.keys():
                    intersection_dict.update({key + '_' + q_type: [index]})
                else:
                    intersection_dict[key + '_' + q_type].append(index)
        intersection_dict_copy = copy.deepcopy(intersection_dict)
        for i in range(len(organ)):
            key = obtain_key(i, intersection_dict_copy)
            random_index = random.randint(0, len(intersection_dict[key]) - 1)
            value = intersection_dict[key].pop(random_index)
            condition_index_list.append(value)

    elif args.use_mix_cond_vq_union:
        v_cond_list = []
        q_cond_list = []
        for index in range(len(question)):
            organ_type = organ[index]
            question_type = question[index].split(",")[0]
            v_type_list[organ_type].append(index)
            q_type_list[question_type].append(index)

        v_type_list_copy = copy.deepcopy(v_type_list)
        for i in range(len(organ)):
            key = obtain_key(i, v_type_list_copy)
            random_index = random.randint(0, len(v_type_list[key]) - 1)
            value = v_type_list[key].pop(random_index)
            v_cond_list.append(value)

        q_type_list_copy = copy.deepcopy(q_type_list)
        for i in range(len(question)):
            key = obtain_key(i, q_type_list_copy)
            random_index = random.randint(0, len(q_type_list[key]) - 1)
            value = q_type_list[key].pop(random_index)
            q_cond_list.append(value)
        for i in range(len(question)):
            if random.random() > 0.5:
                index = v_cond_list[i]
            else:
                index = q_cond_list[i]
            condition_index_list.append(index)

    index = torch.tensor(condition_index_list).cuda()
    mixed_v = [0, 0]
    if args.use_ablation_v:
        mixed_v = v
    else:
        if args.maml:
            mixed_v[0] = lam * v[0] + (1 - lam) * v[0][index, :]
        if args.autoencoder:
            mixed_v[1] = lam * v[1] + (1 - lam) * v[1][index, :]

    a_1, a_2 = a, a[index]
    return mixed_v, a_1, a_2, lam, index


class Trainer(object):
    """
    Main class for training.
    """

    def __init__(self, args, model, criterion, optimizer=None, ae_criterion=None, qc_model=None):
        self.args = args
        # copy model and criterion on current device
        self.model = model.to(self.args.device)
        self.criterion = criterion.to(self.args.device)
        self.ae_criterion = ae_criterion.to(self.args.device)
        if qc_model is not None:
            self.qc_model = qc_model.to(self.args.device)
        self.optimizer = optimizer

    def train_step(self, sample):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get reproducible results
        # torch.manual_seed(self.args.seed)
        # torch.cuda.manual_seed(self.args.seed)

        loss, batch_score = self._forward(sample)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, batch_score

    def _forward(self, sample, eval=False):
        # prepare model and optimizer
        if eval:
            self.model.eval()
        else:
            self.model.train()
        loss = None
        with torch.no_grad() if eval else contextlib.ExitStack():
            answers = sample[2]
            img_data = sample[0][1]
            v_type, q_type = sample[-2]

            if self.args.use_mix:
                if self.args.use_mix_cond:
                    mixed_v, a_1, a_2, lam, index = modality_specific_mixup_data(sample[0], answers, self.args,
                                                                                 (v_type, q_type))
                else:
                    mixed_v, a_1, a_2, lam, index = mixup_data(sample[0], answers, self.args, (v_type, q_type))

                if self.args.autoencoder:
                    features, decoder = self.model(mixed_v, sample[1], lam, index)
                else:
                    features = self.model(mixed_v, sample[1], lam, index)
                preds = self.model.classifier(features)

                if self.args.use_mix_all:
                    loss = mixup_criterion_all(self.criterion, preds.float(), a_1, a_2, lam, self.args.pow)
                elif self.args.use_ablation:
                    loss = mixup_criterion_abla(self.criterion, preds.float(), a_1, a_2, lam)
                else:
                    loss = mixup_criterion(self.criterion, preds.float(), a_1, a_2, lam, self.args.pow)

                if self.args.autoencoder:
                    loss_ae = self.ae_criterion(mixed_v[1], decoder)
                    loss = loss + (loss_ae * self.args.ae_alpha)
            else:
                if self.args.autoencoder:
                    features, decoder = self.model(sample[0], sample[1])
                else:
                    features = self.model(sample[0], sample[1])
                preds = self.model.classifier(features)
                loss = self.criterion(preds.float(), answers)
                if self.args.autoencoder:
                    loss_ae = self.ae_criterion(img_data, decoder)
                    loss = loss + (loss_ae * self.args.ae_alpha)

        batch_score = self._compute_score_with_logits(preds, sample[2].data).sum()
        return loss, batch_score

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _compute_score_with_logits(self, logits, labels):
        if len(labels.shape) == 2:
            logits = torch.max(logits, 1)[1].data  # argmax
            one_hots = torch.zeros(*labels.size()).to(logits.device)
            one_hots.scatter_(1, logits.view(-1, 1), 1)
            scores = (one_hots * labels)
        else:
            logits = torch.max(logits, 1)[1].data
            scores = logits.eq(labels)
        return scores
