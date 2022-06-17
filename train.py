"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import os
import time
import torch
import utils
import torch.nn as nn
from trainer import Trainer
from sklearn.model_selection import  StratifiedKFold, KFold

# Kaiming normalization initialization
def init_weights(m):
    if type(m) == nn.Linear:
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(m.weight)

# VQA score computation
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

# Train phase
def train(args, model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0, qc_model=None):
    device = args.device
    # Scheduler learning rate
    lr_default = args.lr
    lr_decay_step = 3
    lr_decay_rate = .75
    lr_decay_epochs = range(10,30,lr_decay_step) if eval_loader is not None else range(10,30,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 15    # Start point for model saving
    grad_clip = args.clip_norm

    utils.create_dir(output)

    # Adamax optimizer
    # optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
    #     if opt is None else opt
    optim = torch.optim.Adamax(params=model.parameters())

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss() # torch.optim.Adamax
    if args.use_partial_label:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    ae_criterion = torch.nn.MSELoss()

    # write hyper-parameter to log file
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger.write(args.__repr__())
    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    # create trainer
    trainer = Trainer(args, model, criterion, optim, ae_criterion, qc_model)
    update_freq = int(args.update_freq)
    wall_time_start = time.time()

    best_eval_score = 0
    loss_list = []
    eval_loss_list = []
    # Epoch passing in training phase
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        count_norm = 0
        num_updates = 0
        t = time.time()
        N = len(train_loader.dataset)
        num_batches = int(N/args.batch_size + 1)

        for i, (v, q, a, _, q_type, _, _, _, v_type) in enumerate(train_loader):
            if args.maml:
                if 'RAD' in args.RAD_dir:
                    v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                else:
                    v[0] = v[0].reshape(v[0].shape[0], 3, 84, 84)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            q = q.to(device)
            a = a.to(device)
            sample = [v, q, a, (v_type, q_type), epoch]

            loss, batch_score = trainer.train_step(sample)
            count_norm += 1
            total_loss += loss.item()
            train_score += batch_score
            num_updates += 1
            if num_updates % int(args.print_interval / update_freq) == 0:
                print("Iter: {}, Loss {:.6f}, Num updates: {}, Wall time: {:.2f}, ETA: {}".format(i + 1, total_loss / ((num_updates + 1)), num_updates, time.time() - wall_time_start, utils.time_since(t, i / num_batches)))

        total_loss /= num_updates
        train_score = 100 * train_score / (num_updates * args.batch_size)

        # Evaluation
        if eval_loader is not None:
            print("Evaluating...")
            trainer.model.train(False)
            eval_score, bound, eval_loss = evaluate(model, eval_loader, args)
            trainer.model.train(True)

        loss_list.append(str(round(total_loss, 5)))
        eval_loss_list.append(str(round(eval_loss, 5)))
        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.4f, score: %.2f' % (total_loss, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_loader is not None and eval_score > best_eval_score:
            model_path = os.path.join(output, 'model_epoch_best.pth')
            utils.save_model(model_path, model, epoch, trainer.optimizer)
            best_eval_score = eval_score
    logger.write(','.join(loss_list))
    logger.write(','.join(eval_loss_list))

# Evaluation
def evaluate(model, dataloader, args):
    device = args.device
    criterion = torch.nn.BCEWithLogitsLoss()
    score = 0
    upper_bound = 0
    num_data = 0
    loss = 0
    with torch.no_grad():
        for i in iter(dataloader):
            (v, q, a, _, _, p_type, _, _, _) = i
            if args.maml:
                if 'RAD' in args.RAD_dir:
                    if p_type[0] != "freeform":
                        continue
                    v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                else:
                    v[0] = v[0].reshape(v[0].shape[0], 3, 84, 84)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            q = q.to(device)
            a = a.to(device)
            if args.use_grad_cam:
                features, decoder, maml_feat, ae_feat = model(v, q, test=True)
            else:
                if args.autoencoder:
                    features, _ = model(v, q, test=True)
                else:
                    features = model(v, q, test=True)
            preds = model.classifier(features)
            loss += criterion(preds.float(), a).item()

            final_preds = preds
            batch_score = compute_score_with_logits(final_preds, a.data).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += final_preds.size(0)
    if 'RAD' in args.RAD_dir:
        total_count = 308
    else:
        total_count = 6761
    loss = loss / total_count
    score = score / total_count
    upper_bound = upper_bound / total_count
    return score, upper_bound, loss
