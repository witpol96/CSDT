import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
# from datasets.sampler import IdentitySampler
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset

from .sysu import SYSUDataset

import random


__factory = {'SYSU': SYSUDataset}

def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

  

def build_testloader(args, dataset, transforms=None):

    if transforms:
        test_transforms = transforms
    else:
        test_transforms = build_transforms(img_size=args.img_size,
                                            is_train=False)
    num_workers = args.num_workers

    test_img_set = ImageDataset(dataset['image_pids'], dataset['img_paths'],
                                   test_transforms)
    test_txt_set = TextDataset(dataset['caption_pids'],
                                dataset['captions'],
                                text_length=args.text_length)

    test_img_loader = DataLoader(test_img_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    test_txt_loader = DataLoader(test_txt_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=num_workers)    
    
    return test_img_loader, test_txt_loader



def build_trainloader(args, dataset, transforms=None, aug=False):
    logger = logging.getLogger("RDE.dataset")

    num_workers = args.num_workers
    # dataset = __factory[args.dataset_name](root=args.root_dir)
    # num_classes = len(dataset.train_id_container)
    if transforms is None:
        train_transforms = build_transforms(img_size=args.img_size,
                                                aug=False,
                                                is_train=True)
    else:
        train_transforms = transforms
    # aug_transform = build_transforms(img_size=args.img_size,
    #                                         aug=True,
    #                                         is_train=True)

    
    train_set = ImageTextDataset(dataset,args,
                            transform=train_transforms,
                        text_length=args.text_length)

    logger.info('using random sampler')
    train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                collate_fn=collate)
    
    return train_loader