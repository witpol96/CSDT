"""
The main code of Coach class, which supervise the whole training process.
"""
from model import build_model
from datasets.build import build_testloader, build_trainloader, build_transforms, collate
# from torch.utils.data import DataLoader
# from datasets.bases import T2VIDataset
from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import Checkpointer
from utils.metrics import Evaluator, EvaluatorIR, EvaluatorRGB
# import torch
from datasets.sysu import SYSUDataset
from datasets.llcm import LLCMDataset
# from datasets.sampler import GenIdx, IdentitySampler
from model.objectives import DirectionLoss
import torch
import torch.nn.functional as F
factory = {'SYSU': SYSUDataset, "LLCM":LLCMDataset}



class Base:
    def __init__(self, args):
        self.args = args
        self.num_classes = 395
        self._init_dataloader()
        self._init_model()
        self._init_optimizer()
        self._init_checkpointer()

        self.compute_direction_loss = DirectionLoss()

    def _init_model(self):
        # TODO Init RDE model
        # self.model = Model(self.pid_num, self.img_h, self.img_w).to(self.device)
        self.model = build_model(self.args, self.num_classes)
        

    def _init_dataloader(self):
        dataset = factory[self.args.dataset_name](root=self.args.root_dir)
        self.dataset = dataset
        self.num_classes = dataset.train_id_container
        print(f"Build testloader for text2rgb task, {len(dataset.test_rgb['image_pids'])} samples")
        self.val_img_loader, self.val_txt_loader = build_testloader(self.args, dataset.test_rgb)

        print(f"Build testloader for text2ir task, {len(dataset.test_ir['image_pids'])} samples")
        self.val_img_loader2, self.val_txt_loader2 = build_testloader(self.args, dataset.test_ir)

        # print(f"Build trainloader for text2rgb task, {len(dataset.train_rgb)} samples")
        # self.trainloader_rgb  = build_trainloader(self.args, dataset.train_rgb)

        # print(f"Build trainloader for text2ir task, {len(dataset.train_ir)} samples")
        # aug_transforms =  build_transforms(img_size=self.args.img_size,
        #                                         aug=True,
        #                                         is_train=True)
        # self.trainloader_ir  = build_trainloader(self.args, dataset.train_rgb + dataset.train_ir, transforms=aug_transforms)
        # self.trainloader_ir  = build_trainloader(self.args, dataset.train_ir)

        print(f"Build trainloader for text2multi task, {len(dataset.train_rgb)} {len(dataset.train_ir)} samples")
        self.trainloader_multi  = build_trainloader(self.args, dataset.train_rgb + dataset.train_ir)
    
        self.evaluator_ir = EvaluatorIR(self.val_img_loader2, self.val_txt_loader2)
        self.evaluator_rgb = EvaluatorRGB(self.val_img_loader, self.val_txt_loader)

    def _init_checkpointer(self):
        self.checkpointer = Checkpointer(self.model, self.optimizer, self.scheduler, self.args.output_dir,self.args.output_dir)


    # def _init_optimizer_stage1(self):
    #     self.optimizer_stage1 = build_optimizer_stage1(self.args, self.model)
    #     self.scheduler_stage1 = build_lr_scheduler(self.args, self.optimizer_stage1)


    def compute_distil_loss(self, t_feats1, t_feats2, i_feats1, i_feats2):
        t_feats1 = F.normalize(t_feats1, dim=-1)
        t_feats2 = F.normalize(t_feats2, dim=-1)
        i_feats1 = F.normalize(i_feats1, dim=-1)
        i_feats2 = F.normalize(i_feats2, dim=-1)

        logits_rgb = t_feats1 @ i_feats1.T   # [B, B]
        logits_ir  = t_feats2 @ i_feats2.T   # [B, B]
        T = 0.02
        p_rgb = F.softmax(logits_rgb / T, dim=1).detach()
        log_p_ir = F.log_softmax(logits_ir / T, dim=1)
        loss = F.kl_div(log_p_ir, p_rgb, reduction='batchmean') * (T * T)
        return loss

    def compute_sdm(self, t_feats1, t_feats2, i_feats1, i_feats2, pid, epsilon=1e-8):
        """
        Similarity Distribution Matching
        """
        batch_size = t_feats1.shape[0]
        pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()
        labels_distribute = labels / labels.sum(dim=1)
        
        logit_scale = torch.ones([]) * (1 / self.args.temperature) 
        t2rgb_cosine_theta = t_feats1 @ i_feats1.T 
        rgb2t_cosine_theta = t2rgb_cosine_theta.t()

        t2ir_cosine_theta = t_feats2 @ i_feats2.T 
        ir2t_cosine_theta = t2ir_cosine_theta.t()

        text_proj_rgb = logit_scale * t2rgb_cosine_theta
        rgb_proj_text = logit_scale * rgb2t_cosine_theta

        text_proj_ir = logit_scale * t2ir_cosine_theta
        ir_proj_text = logit_scale * ir2t_cosine_theta

        rgb2t_pred = F.softmax(rgb_proj_text, dim=1)
        t2rgb_pred = F.softmax(text_proj_rgb, dim=1)
        ir2t_pred = F.softmax(ir_proj_text, dim=1)
        t2ir_pred = F.softmax(text_proj_ir, dim=1)


        i2t_loss = ir2t_pred * (F.log_softmax(ir_proj_text, dim=1) - torch.log(0.9*rgb2t_pred.detach() + 0.1*labels_distribute + epsilon))
        t2i_loss = t2ir_pred * (F.log_softmax(text_proj_ir, dim=1) - torch.log(0.9*t2rgb_pred.detach() + 0.1*labels_distribute + epsilon))

        loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)
        # loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
# 
        return loss

    def _init_optimizer(self):
        self.optimizer = build_optimizer(self.args, self.model)
        self.scheduler = build_lr_scheduler(self.args, self.optimizer)

    # def _freeze_stage1_model(self):
    #     print('freezing stage1 model ...')
    #     for name, param in self.model.named_parameters():
    #         param.requires_grad = False

    # def _init_t2vi_dataset(self):
    #     self.color_pos, self.thermal_pos = GenIdx(self.dataset.train_rgb, self.dataset.train_ir)
    #     train_transforms = build_transforms(img_size=self.args.img_size,
    #                                         aug=False,
    #                                         is_train=True)
    #     self.trainset_t2vi = T2VIDataset(self.dataset.train_rgb,
    #                                 self.dataset.train_ir,
    #                                 transform=train_transforms,
    #                                 text_length=self.args.text_length)
          