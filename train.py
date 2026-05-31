import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from torch.nn import functional as F
from model import objectives
import torch.nn as nn
import copy
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize



################### CODE FOR THE BETA MODEL  ########################

import scipy.stats as stats
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta
    

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)

def get_loss_ir(model, data_loader, args, bound=True):
    logger = logging.getLogger("RDE.train")
    model.eval()
    device = args.device
    data_size = data_loader.dataset.__len__()
    # real_labels = data_loader.dataset.real_correspondences
    lossA, lossB, simsA, simsB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size),torch.zeros(data_size)
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch['index']
        with torch.no_grad():
            la, lb, sa, sb = model.compute_per_loss_ir(batch, bound)
            for b in range(la.size(0)):
                lossA[index[b]]= la[b]
                lossB[index[b]]= lb[b]
                simsA[index[b]]= sa[b]
                simsB[index[b]]= sb[b]
            if i % 100 == 0:
                logger.info(f'compute loss batch {i}')

    losses_A = (lossA-lossA.min())/(lossA.max()-lossA.min())    
    losses_B = (lossB-lossB.min())/(lossB.max()-lossB.min())
    
    input_loss_A = losses_A.reshape(-1,1) 
    input_loss_B = losses_B.reshape(-1,1)
    

    logger.info('\nFitting GMM ...') 
 
    if model.args.noisy_rate > 0.4 or model.args.dataset_name=='RSTPReid':
    #     # should have a better fit 
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]
    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
    # ------------------------------------------------------------------
    # loss_data = input_loss_B.numpy()
    # plt.figure(figsize=(8, 6))
    # plt.hist(loss_data, bins=400, color='skyblue', edgecolor='black')  
    # plt.title("Histogram of Tensor Values")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")

    # plt.savefig(f"tensor_histogram_wobound.png")
    # print("histogram has been saved")
 
    # -----------------------------------------------------------------------------
    pred_A = split_prob(prob_A, 0.5)
    pred_B = split_prob(prob_B, 0.5)
  
    return torch.Tensor(pred_A), torch.Tensor(pred_B)


def train(args, coach):
    # initialize train stage 2
    model = coach.model.to(args.device)
    train_loader = coach.trainloader_multi
    evaluator = coach.evaluator_rgb
    evaluator2 = coach.evaluator_ir
    optimizer = coach.optimizer
    scheduler = coach.scheduler
    checkpointer = coach.checkpointer



    log_period = args.log_period
    eval_period = args.eval_period
    device = args.device
    num_epoch = 6 if args.dataset_name == 'LLCM' else 10
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("RDE.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "loss1": AverageMeter(),
        "loss2": AverageMeter(),
        "loss3": AverageMeter(),
        "loss4": AverageMeter(),
        "loss5": AverageMeter(),
        "loss6": AverageMeter(),
        "loss7": AverageMeter(),
        "bge_loss2": AverageMeter(),
        "tse_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "direction_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0
    best_epoch = 0
    # ensure output dir exists for saving best model
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception:
        pass
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    
    # for k, v in model.named_parameters():
    #     print(k)

    checkpointer.resume(f="/data1/zza_data/reid/pretrained/best0.pth")
    # model.reset_vision_encoder()
    print("Start the 2nd stage of training")

    def _update_trainable(model, optimizer, epoch, freeze_epochs=5):
        # Freeze `base_model` for the first `freeze_epochs` epochs.
        # After that, unfreeze all parameters so the entire model is trainable.
        if epoch <= freeze_epochs:
            for name, param in model.named_parameters():
                if 'base_model' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
        # Do not modify optimizer.param_groups here; optimizers skip parameters without gradients.

    for epoch in range(1, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        model.epoch = epoch
        _update_trainable(model, optimizer, epoch, freeze_epochs=1)

        pred_A, pred_B = get_loss_ir(model, train_loader, args, bound=True)
        consensus_division = pred_A + pred_B # 0,1,2 
        if args.dataset_name == 'SYSU':
            consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),))
            label_hat = consensus_division.clone()
            label_hat[consensus_division>1] = 1
            label_hat[consensus_division<=1] = 0 
        else:
            label_hat = consensus_division.clone()
            label_hat[consensus_division>=1] = 1
            label_hat[consensus_division<1] = 0 

        label_hat_rgb = label_hat

        print(label_hat.sum(), label_hat_rgb.sum())

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            
            batch['label_hat'] = label_hat[index.cpu()]
            batch['label_hat_rgb'] = label_hat_rgb[index.cpu()]
 
            images = batch["images"]
            imgtype = batch["image_type"]
            caption_ids = batch["caption_ids"]

            if images[imgtype==1].shape[0] == 0 or images[imgtype==0].shape[0] == 0:
                continue
            

            image_feats_ir, atten_i_ir = model.encode_ir_image(images, imgtype, ca=True)
            image_feats, atten_i = model.encode_rgb_image(images[imgtype==1])
            text_feats_ir, atten_t_ir = model.encode_ir_text(caption_ids)
            text_feats_rgb, atten_t_rgb = model.encode_rgb_text(caption_ids)

            
            i_feats_rgb = image_feats[:, 0, :].float()
            i_feats_ir = image_feats_ir[:, 0, :].float()
            
            t_feats_rgb = text_feats_rgb[torch.arange(text_feats_rgb.shape[0]), caption_ids.argmax(dim=-1)].float()
            t_feats_ir = text_feats_ir[torch.arange(text_feats_ir.shape[0]), caption_ids.argmax(dim=-1)].float()

            i_tse_f_rgb = model.ir_visul_emb_layer(image_feats, atten_i, 0)
            i_tse_f_ir = model.ir_visul_emb_layer(image_feats_ir, atten_i_ir, 0)
            t_tse_f_rgb = model.ir_texual_emb_layer(text_feats_rgb, caption_ids, atten_t_rgb)
            t_tse_f_ir = model.ir_texual_emb_layer(text_feats_ir, caption_ids, atten_t_ir)
  
            label_hat_batch = batch['label_hat'].to(i_feats_ir.device) 
            label_hat_batch_rgb = batch['label_hat_rgb'].to(i_feats_rgb.device) 

            loss1, loss2 = objectives.compute_rbs(i_feats_rgb, t_feats_rgb[imgtype==1], i_tse_f_rgb, t_tse_f_rgb[imgtype==1], batch['pids'][imgtype==1], \
                                                label_hat=label_hat_batch_rgb[imgtype==1], margin=model.args.margin,tau=model.args.tau,\
                                                    loss_type=model.loss_type,logit_scale=model.logit_scale)



            loss3, loss4 = objectives.compute_rbs(i_feats_ir, t_feats_ir, i_tse_f_ir, t_tse_f_ir, batch['pids'], \
                                                label_hat=label_hat_batch * (imgtype==0 + 0), margin=model.args.margin,tau=model.args.tau,\
                                                    loss_type=model.loss_type,logit_scale=model.logit_scale)

            # i_tse_f_rgb_n = i_tse_f_rgb / i_tse_f_rgb.norm(dim=-1, keepdim=True)
            # i_tse_f_ir_n = i_tse_f_ir[imgtype==1] / i_tse_f_ir[imgtype==1].norm(dim=-1, keepdim=True)
            # t_tse_f_rgb_n = t_tse_f_rgb[imgtype==1] / t_tse_f_rgb[imgtype==1].norm(dim=-1, keepdim=True)
            # t_tse_f_ir_n = t_tse_f_ir[imgtype==1] / t_tse_f_ir[imgtype==1].norm(dim=-1, keepdim=True)

            # loss5 = coach.compute_direction_loss((i_tse_f_rgb_n - i_tse_f_ir_n).detach(), t_tse_f_rgb_n-t_tse_f_ir_n)
            # loss5 = (label_hat_batch_rgb[imgtype==1] * loss5).sum() / label_hat_batch_rgb[imgtype==1].sum()

            i_feats_irn = i_feats_ir[imgtype==1] / i_feats_ir[imgtype==1].norm(dim=-1, keepdim=True)
            i_feats_rgbn = i_feats_rgb / i_feats_rgb.norm(dim=-1, keepdim=True)

            t_feats_irn = t_feats_ir[imgtype==1] / t_feats_ir[imgtype==1].norm(dim=-1, keepdim=True)
            t_feats_rgbn = t_feats_rgb[imgtype==1] / t_feats_rgb[imgtype==1].norm(dim=-1, keepdim=True)
            
            i_tse_f_rgb_n = i_tse_f_rgb / i_tse_f_rgb.norm(dim=-1, keepdim=True)
            i_tse_f_ir_n = i_tse_f_ir[imgtype==1] / i_tse_f_ir[imgtype==1].norm(dim=-1, keepdim=True)
            t_tse_f_rgb_n = t_tse_f_rgb[imgtype==1] / t_tse_f_rgb[imgtype==1].norm(dim=-1, keepdim=True)
            t_tse_f_ir_n = t_tse_f_ir[imgtype==1] / t_tse_f_ir[imgtype==1].norm(dim=-1, keepdim=True)

            i_local_rgb = image_feats[:, 1:, :].mean(dim=1).float()
            i_local_ir = image_feats_ir[:, 1:, :].mean(dim=1).float()

            i_local_rgbn = i_local_rgb / i_local_rgb.norm(dim=-1, keepdim=True)
            i_local_irn = i_local_ir[imgtype==1] / i_local_ir[imgtype==1].norm(dim=-1, keepdim=True)
            
            loss5 = coach.compute_sdm(t_feats_rgbn, t_feats_irn, i_feats_rgbn, i_feats_irn, batch['pids'][imgtype==1])
            loss5 = (label_hat_batch_rgb[imgtype==1] * loss5).sum() / label_hat_batch_rgb[imgtype==1].sum()
            loss6 = coach.compute_sdm(t_feats_rgbn, t_tse_f_ir_n, i_feats_rgbn, i_tse_f_ir_n, batch['pids'][imgtype==1])
            loss6 = (label_hat_batch_rgb[imgtype==1] * loss6).sum() / label_hat_batch_rgb[imgtype==1].sum()

            loss7 = 1 - coach.compute_direction_loss(i_local_irn,  i_local_rgbn)
            loss7 = (label_hat_batch_rgb[imgtype==1] * loss7).sum() / label_hat_batch_rgb[imgtype==1].sum()


            if args.dataset_name == 'SYSU':
                total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
            elif args.dataset_name == 'LLCM':
                total_loss = 0.1 * (loss1 + loss2 + loss5 + loss6 + loss7) + 0.5 * (loss3+loss4) 
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['loss1'].update(loss1, batch_size)
            # meters['loss2'].update(loss2, batch_size)
            meters['loss3'].update(loss3, batch_size)
            # meters['loss4'].update(loss4, batch_size)
            meters['direction_loss'].update(loss5, batch_size)
            meters['loss6'].update(loss6, batch_size)
            meters['loss7'].update(loss7, batch_size)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)       
 
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg >= 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
            
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1_rgb = evaluator.eval(model.eval())
                top1 = evaluator2.eval(model.eval())
                # top2 = evaluator3.eval(model.eval())
                top1 = (top1_rgb + top1) / 2
                if best_top1 < top1:
                    best_top1 = top1
                    best_epoch = epoch
                    # save best model (only on rank 0)
                    if get_rank() == 0:
                        try:
                            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'top1': float(best_top1)}, best_model_path)
                            logger.info(f"Saved best model to {best_model_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save best model: {e}")
                    # checkpointer.save("best", **arguments)
        
    # end for epoch

    # After training, load best model from disk (if exists) and run evaluators
    if get_rank() == 0:
        if os.path.exists(best_model_path):
            try:
                ckpt = torch.load(best_model_path, map_location=device)
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                    saved_epoch = ckpt.get('epoch', 'unknown')
                else:
                    model.load_state_dict(ckpt)
                    saved_epoch = 'unknown'
                model.to(device)
                model.eval()
                logger.info(f"Loaded best model from {best_model_path} (epoch={saved_epoch})")
                logger.info("--- Evaluator RGB (best model) ---")
                evaluator.eval(model)
                logger.info("--- Evaluator IR (best model) ---")
                evaluator2.eval(model)
            except Exception as e:
                logger.warning(f"Failed to load/evaluate best model: {e}")
        else:
            logger.info("No best model file found; skipping final evaluation.")
    

