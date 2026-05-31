# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
from collections import OrderedDict

import torch
import math


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        # if not self.save_to_disk:
        #     return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def resume(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            raise IOError(f"No Checkpoint file found on {f}")
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        # if "optimizer" in checkpoint and self.optimizer:
        #     self.logger.info("Loading optimizer from {}".format(f))
        #     self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        # if "scheduler" in checkpoint and self.scheduler:
        #     self.logger.info("Loading scheduler from {}".format(f))
        #     self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        # return any further checkpoint data
        return checkpoint

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, except_keys=None):
        load_state_dict(self.model, checkpoint.pop("model"), except_keys)


def check_key(key, except_keys):
    if except_keys is None:
        return False
    else:
        for except_key in except_keys:
            if except_key in key:
                return True
        return False


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger("PersonSearch.checkpoint")
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if check_key(key, except_keys):
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, except_keys=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys)

    # use strict loading
    model.load_state_dict(model_state_dict)
    # load_param(model, model_state_dict)

def load_param(model, state_dict):
    # 将pretrained_dict里不属于model_dict的键剔除掉
    # for k, _ in state_dict.items():
    #     print(k)
    param_dict =  {k: v for k, v in state_dict.items() if k in model.state_dict()}
    # print(len(param_dict.keys()), len(model.state_dict().keys()))

    for k, v in param_dict.items():
        if k == 'base_model.visual.positional_embedding' and v.shape != model.base_model.visual.positional_embedding.shape:
            v = resize_pos_embed(v, model.base_model.visual.positional_embedding, model.base_model.visual.num_y, model.base_model.visual.num_x)
        try:
            model.state_dict()[k].copy_(v)
        except:
            print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
            print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
    print([k for k in model.state_dict().keys() if "base_model.visual.class_embedding" in k])
    model.state_dict()["ir_token_emb.weight"].copy_(param_dict["base_model.token_embedding.weight"])
    model.state_dict()["rgb_token_emb.weight"].copy_(param_dict["base_model.token_embedding.weight"])
    model.state_dict()["ir_shallow.weight"].copy_(param_dict["base_model.visual.conv1.weight"])
    model.state_dict()["rgb_shallow.weight"].copy_(param_dict["base_model.visual.conv1.weight"])
    model.state_dict()["ir_vproj"].copy_(param_dict["base_model.visual.proj"])
    model.state_dict()["rgb_vproj"].copy_(param_dict["base_model.visual.proj"])
    model.state_dict()["ir_tproj"].copy_(param_dict["base_model.text_projection"])
    model.state_dict()["rgb_tproj"].copy_(param_dict["base_model.text_projection"])
    model.state_dict()["ir_cls_emb"].copy_(param_dict["base_model.visual.class_embedding"])
    model.state_dict()["rgb_cls_emb"].copy_(param_dict["base_model.visual.class_embedding"])


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)
    # print(posemb.shape, posemb_new.shape)
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[0, :1], posemb[0, 1:]

    # gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    # posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    # posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    # posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    # print(posemb_token.shape, posemb_grid.shape)
    posemb = torch.cat([posemb_token, posemb_token, posemb_grid], dim=0)
    # print(posemb.shape)
    return posemb.squeeze(0)

