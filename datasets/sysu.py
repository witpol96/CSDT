import os as os
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset
import json
import torch
import random

class SYSUDataset(BaseDataset):
    dataset_dir = 'SYSU-MM01'
    # dataset_dir = "RegDB"
    def __init__(self, root='', verbose=True):
        super(SYSUDataset, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.img_dir = self.dataset_dir
        self.anno_path = os.path.join(self.dataset_dir, 'data_captions.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
        self.train_annos.extend(self.val_annos)

        # dual spectrum datasets
        self.train_rgb, self.train_ir, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test_rgb, self.test_ir, self.test_id_container = self._process_anno(self.test_annos)

        if verbose:
            self.logger.info("=> SYSU-MM01 Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos
    

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        # id2pid = []
        id2pid = dict()
        if training:
            dataset_rgb = []
            dataset_ir = []
            rgb_image_id = 0
            ir_image_id = 0
            for anno in annos:
                id = anno['id']
                pid = id2pid.get(id,len(pid_container))
                id2pid[id] = pid
                pid_container.add(pid)
                img_path = os.path.join(self.img_dir, anno['img_path'])
                captions = anno['captions'] 
                if anno['type'] == "visible":
                    for caption in captions:
                        dataset_rgb.append((pid, rgb_image_id, img_path, caption, 1))
                    rgb_image_id += 1
                else:
                    for caption in captions:
                        dataset_ir.append((pid, ir_image_id, img_path, caption, 0))
                    ir_image_id += 1
               
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset_rgb,dataset_ir,pid_container# make pid begin from 0
        else:
            dataset_rgb = {}
            dataset_ir = {}
            img_paths1 = []
            captions1 = []
            image_pids1 = []
            caption_pids1 = []
            img_paths2 = []
            captions2 = []
            image_pids2 = []
            caption_pids2 = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = os.path.join(self.img_dir, anno['img_path'])
                if anno["type"] == "visible":
                    img_paths1.append(img_path)
                    image_pids1.append(pid)
                    caption_list = anno['captions'] # caption list
                    for caption in caption_list:
                        captions1.append(caption)
                        caption_pids1.append(pid)
                else:
                    img_paths2.append(img_path)
                    image_pids2.append(pid)
                    caption_list = anno['captions'] # caption list
                    for caption in caption_list:
                        captions2.append(caption)
                        caption_pids2.append(pid)

            dataset_rgb = {
                "image_pids": image_pids1,
                "img_paths": img_paths1,
                "caption_pids": caption_pids1,
                "captions": captions1
            }

            dataset_ir = {
                "image_pids": image_pids2,
                "img_paths": img_paths2,
                "caption_pids": caption_pids2,
                "captions": captions2
            }
            return dataset_rgb, dataset_ir, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not os.path.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
