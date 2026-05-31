from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer, l2norm
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights,LayerNorm
import torch
import torch.nn as nn 
import torch.nn.functional as F
from datasets.bases import tokenize
from utils.simple_tokenizer import SimpleTokenizer
import copy
import random
# from .encoder_wlora import *

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 


        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # cls_emb = torch.stack([copy.deepcopy(self.base_model.visual.class_embedding),copy.deepcopy(self.base_model.visual.class_embedding)], dim=0).to(args.device)
        # self.cls_emb = nn.Parameter(cls_emb)
        # ---------------------------------------------------------------------
        # eos_emb = torch.empty(2, 1, 512).to(args.device)
        # nn.init.normal_(eos_emb, std=0.02)
        # self.eos_emb = nn.Parameter(eos_emb)
        # ---------------------------------------------------------------------
        # eos_emb = self.base_model.token_embedding(tokenize("",SimpleTokenizer()))[1].unsqueeze(0).unsqueeze(0).expand(2,-1,-1)
        # self.eos_emb = nn.Parameter(eos_emb)
        # ----------------------------------------------------------------------
        self.ir_cls_emb = copy.deepcopy(self.base_model.visual.class_embedding)
        self.rgb_cls_emb = copy.deepcopy(self.base_model.visual.class_embedding)
        self.ir_token_emb = copy.deepcopy(self.base_model.token_embedding)
        self.rgb_token_emb = copy.deepcopy(self.base_model.token_embedding)

        # self.vision_encoder = self.build_vision_encoder(self.base_model)
        # self.text_encoder = self.build_text_encoder(self.base_model)

        # ir_eos_emb = torch.empty(1, 512).to(args.device)
        # nn.init.normal_(ir_eos_emb, std=0.02)
        # self.ir_eos_emb = nn.Parameter(ir_eos_emb)

        # rgb_eos_emb = torch.empty(1, 512).to(args.device)
        # nn.init.normal_(rgb_eos_emb, std=0.02)
        # self.rgb_eos_emb = nn.Parameter(rgb_eos_emb)
        
        self.ir_shallow = copy.deepcopy(self.base_model.visual.conv1)
        self.rgb_shallow = copy.deepcopy(self.base_model.visual.conv1)

        self.ir_vproj = copy.deepcopy(self.base_model.visual.proj)
        self.rgb_vproj = copy.deepcopy(self.base_model.visual.proj)

        self.ir_tproj = copy.deepcopy(self.base_model.text_projection)
        self.rgb_tproj = copy.deepcopy(self.base_model.text_projection)

        self.rgb_visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.ir_visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.rgb_texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        self.ir_texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        # ----------------------------------------------------------------------------------
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type

 

    # def build_vision_encoder(self,base_model):
    #     # if args.add_lora:
    #     vision_width = 768
    #     vision_layers = 12
    #     vision_heads = 12
    #     transformer = MMTransformer_withlora(width=vision_width,layers=vision_layers,heads=vision_heads,lora_r=4, num_loras=2,lora_layers=0)
    #     print('Pretrained Multimodal Encoder with LoRAs Loaded, with LoRA_r={}, LoRA_layers={}'.format(4,2))
    #     return transformer
    
    # def build_text_encoder(self,base_model):
    #     # if args.add_lora:
    #     transformer_width = 512
    #     transformer_layers = 12
    #     transformer_heads = 8
    #     transformer = MMTransformer_withlora(width=transformer_width,layers=transformer_layers,heads=transformer_heads,attn_mask=self.base_model.build_attention_mask(),lora_r=4, num_loras=2,lora_layers=2)
    #     print('Pretrained Multimodal Encoder with LoRAs Loaded, with LoRA_r={}, LoRA_layers={}'.format(4,2))
    #     return transformer
    
    
    # def reset_vision_encoder(self):
    #     stat = copy.deepcopy(self.base_model.visual.transformer).state_dict()
    #     self.vision_encoder.load_state_dict(stat,strict=False)
    #     stat2 = copy.deepcopy(self.base_model.transformer).state_dict()
    #     self.text_encoder.load_state_dict(stat2,strict=False)

    
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')


    # -------------------------------------------------------------------
    def encode_ir_image(self, x, imgtype=None, ca=False):
        if ca:
            idx = random.randint(0, 2)
            if imgtype is not None:
                if idx == 0:
                    x[imgtype==1,1, :, :] = x[imgtype==1, 0, :, :]
                    x[imgtype==1,2, :, :] = x[imgtype==1, 0, :, :]
                elif idx == 1:
                    x[imgtype==1,0, :, :] = x[imgtype==1, 1, :, :]
                    x[imgtype==1,2, :, :] = x[imgtype==1, 1, :, :]
                elif idx == 2:
                    x[imgtype==1,0, :, :] = x[imgtype==1, 2, :, :]
                    x[imgtype==1,1, :, :] = x[imgtype==1, 2, :, :]
            else:
                if idx == 0:
                    x[:,1, :, :] = x[:, 0, :, :]
                    x[:,2, :, :] = x[:, 0, :, :]
                elif idx == 1:
                    x[:,0, :, :] = x[:, 1, :, :]
                    x[:,2, :, :] = x[:, 1, :, :]
                elif idx == 2:
                    x[:,0, :, :] = x[:, 2, :, :]
                    x[:,1, :, :] = x[:, 2, :, :]

        x = self.ir_shallow(x.type(self.base_model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat([
            self.ir_cls_emb.type(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        x = x + self.base_model.visual.positional_embedding.to(x.dtype)
        x = self.base_model.visual.ln_pre(x)
    
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.base_model.visual.transformer([x])
        # outputs = self.vision_encoder([x], 0)
        # outputs = [x]
        # for i, resblock in enumerate(self.base_model.visual.transformer.resblocks):
        #     outputs = resblock(outputs) 
        #     outputs[0][0, :, :] = outputs[0][0, :, :] - outputs[0][1, :, :]
        
        x = outputs[0]
        atten = outputs[1]
        # print(type(atten))
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.base_model.visual.ln_post(x)

        x = x @ self.ir_vproj
        return x,atten
    
    def encode_rgb_image(self, x):
        x = self.rgb_shallow(x.type(self.base_model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat([
            self.base_model.visual.class_embedding.type(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        x = x + self.base_model.visual.positional_embedding.to(x.dtype)
        x = self.base_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.base_model.visual.transformer([x])
        # outputs = self.vision_encoder([x], 1)
        # outputs = [x]
        # for i, resblock in enumerate(self.base_model.visual.transformer.resblocks):
        # outputs = self.vision_encoder([x], 1)
        # for i, resblock in enumerate(self.vision_encoder.resblocks):
        #     outputs = resblock(outputs, 1) 
            # outputs[0][0, :, :] = outputs[0][0, :, :] - outputs[0][1, :, :]
            # print(type(outputs[1]), i)

        x = outputs[0]
        atten = outputs[1]
        # print(type(atten))
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.base_model.visual.ln_post(x)

        x = x @ self.ir_vproj
        return x,atten

    def encode_ir_text(self, text):
        x = self.ir_token_emb(text).type(self.base_model.dtype)  # [batch_size, n_ctx, d_model]
        # emb_list = []
        # eos_emb = self.ir_eos_emb.half()
        # for i in range(x.shape[0]):
        #     emb = torch.cat([
        #         x[i,:text.argmax(dim=-1)[i]], eos_emb, x[i,text.argmax(dim=-1)[i]+1:]
        #     ])
        #     emb_list.append(emb)
        # x = torch.stack(emb_list,dim=0)
        x = x + self.base_model.positional_embedding.type(self.base_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.base_model.transformer([x])
        # outputs = self.text_encoder([x], 0)
        x = outputs[0]
        atten = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.base_model.ln_final(x).type(self.base_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.ir_tproj

        return x,atten
    
    def encode_rgb_text(self, text):
        x = self.rgb_token_emb(text).type(self.base_model.dtype)  # [batch_size, n_ctx, d_model]
        # emb_list = []
        # eos_emb = self.rgb_eos_emb.half()
        # for i in range(x.shape[0]):
        #     emb = torch.cat([
        #         x[i,:text.argmax(dim=-1)[i]], eos_emb, x[i,text.argmax(dim=-1)[i]+1:]
        #     ])
        #     emb_list.append(emb)
        # x = torch.stack(emb_list,dim=0)
        x = x + self.base_model.positional_embedding.type(self.base_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.base_model.transformer([x])
        # outputs = self.text_encoder([x], 1)
        x = outputs[0]
        atten = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.base_model.ln_final(x).type(self.base_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.ir_tproj

        return x,atten

    # -------------------------------------------------------------------    

    def compute_per_loss_ir(self, batch, bound=True):
        images = batch['images']
        caption_ids = batch['caption_ids']
        imgtype = batch['image_type']
        # image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        image_feats, atten_i = self.encode_ir_image(images, imgtype, True)

        text_feats, atten_t = self.encode_ir_text(caption_ids)

        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.ir_visul_emb_layer(image_feats, atten_i, 0)
        t_tse_f = self.ir_texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale, bound=bound)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale, bound=bound)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB


    def compute_per_loss_rgb(self, batch, bound=True):
        images = batch['images']
        caption_ids = batch['caption_ids']
        # imgtype = batch['image_type']
        # image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        image_feats, atten_i = self.encode_rgb_image(images)

        text_feats, atten_t = self.encode_rgb_text(caption_ids)

        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.rgb_visul_emb_layer(image_feats, atten_i, 0)
        t_tse_f = self.rgb_texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale, bound=bound)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale, bound=bound)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB




    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})
  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
