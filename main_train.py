import os

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

import argparse
import torch
import time
import cv2
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import os.path as osp

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler.cosine_lr import CosineLRScheduler
from transformers import AutoTokenizer, RobertaTokenizer
from omegaconf import OmegaConf

from datasets.dataset_GID import Dataset_GID_train, collate_fn
from models_train.net import Net
from utils.misc import get_grad_norm, build_dataset_class_lists
from utils.metrics import Evaluator


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def reduce_value(value, world_size, avg=True):
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size if avg else value


def make_img_gird(img_input, col):
    '''
    :param img_tensor: numpy array, B × H × W × 3
    :param col: the column of the gird
    :return:
    '''
    out_list = []
    k = 0
    while True:
        row_list = []
        for i in range(col):
            row_list.append(img_input[k])
            k += 1
            if k == img_input.shape[0]:
                break
        out_list.append(np.concatenate(row_list, axis=1))
        if k == img_input.shape[0]:
            break
    out = np.concatenate(out_list, axis=0)
    return out


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


class Trainer():
    def __init__(self, cfg, rank):
        self.cfg = cfg
        self.rank = rank

        # model
        self.net = Net(cfg).to(rank)
        self.net = DDP(self.net, device_ids=[rank], find_unused_parameters=True)

        # Optimizer
        parameters = set_weight_decay(self.net, {}, {})
        self.optimizer = torch.optim.AdamW(
            parameters,
            eps=cfg.train.optimizer.eps,
            betas=cfg.train.optimizer.betas,
            lr=cfg.train.base_lr,
            weight_decay=cfg.train.weight_decay)

        self.metric = Evaluator(num_class=6)
        self.lr_scheduler = None

    def update(self, image, text, cross_image=None, cross_entity=None):
        self.optimizer.zero_grad()
        loss_dict = self.net(image=image, text=text, cross_image=cross_image, cross_entity=cross_entity)
        total_loss = loss_dict['loss']
        total_loss.backward()
        if self.cfg.train.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.train.clip_grad)
        else:
            grad_norm = get_grad_norm(self.net.parameters())
        self.optimizer.step()
        return loss_dict

    def save(self):
        if self.rank == 0:
            torch.save(self.net.module.state_dict(), r'checkpoint/net.pt')
            torch.save(self.optimizer.state_dict(), r'checkpoint/opt.pt')
            torch.save(self.lr_scheduler.state_dict(), r'checkpoint/lr_scheduler.pt')
            # torch.save(self.scaler.state_dict(), r'checkpoint/scaler.pt')
            print('save')

    def val_save(self, iter, accu_val):
        if self.rank == 0:
            torch.save(self.net.module.state_dict(), r'checkpoint/net_' + str(iter) + '_' + str(accu_val)
                       + '.pt')
            torch.save(self.optimizer.state_dict(), r'checkpoint/opt_' + str(iter) + '_' + str(accu_val)
                       + '.pt')
            torch.save(self.lr_scheduler.state_dict(), r'checkpoint/lr_scheduler_' + str(iter) + '_' + str(accu_val)
                       + '.pt')
            # torch.save(self.scaler.state_dict(), r'checkpoint/scaler_' + str(iter) + '_' + str(accu_val)
            #            + '.pt')

    def resume(self, epoch):
        last_model_name = r'checkpoint/net.pt'
        state_dict = torch.load(last_model_name)
        self.net.module.load_state_dict(state_dict)

        last_opt_name = r'checkpoint/opt.pt'
        state_dict = torch.load(last_opt_name)
        self.optimizer.load_state_dict(state_dict)

        last_lr_scheduler_name = r'checkpoint/lr_scheduler.pt'
        state_dict = torch.load(last_lr_scheduler_name)
        self.lr_scheduler.load_state_dict(state_dict)

        print('Resume from iteration %d' % epoch)

    def build_scheduler(self, n_iter_per_epoch):
        num_steps = int(self.cfg.train.epochs * n_iter_per_epoch)
        warmup_steps = int(self.cfg.train.warmup_epochs * n_iter_per_epoch)
        if self.cfg.train.lr_scheduler.name == 'cosine':
            lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=num_steps,
                # t_mul=1.,  ## this does not work with higher versions of timm
                lr_min=self.cfg.train.min_lr,
                warmup_lr_init=self.cfg.train.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )
            self.lr_scheduler = lr_scheduler
        else:
            raise NotImplementedError(f'lr scheduler {self.cfg.train.lr_scheduler.name} not implemented')

    def show_result(self, input, tokenizer, epoch, idx):
        self.net.eval()
        COLOR_LIST = np.loadtxt('models_coarse/group_palette.txt', dtype=np.uint8)[:, ::-1]

        CLASS_TEXTS = ['building', 'farmland', 'forest', 'meadow', 'water']

        text_prompt = build_dataset_class_lists(template_set='simple', classnames=CLASS_TEXTS)
        infer_tokens = tokenizer(text_prompt, return_tensors='pt', padding='max_length',
                                 truncation=True, max_length=77)
        infer_tokens = {key: val.cuda() for key, val in infer_tokens.items()}
        imgs = input['image'].cuda()

        result = self.net.module.inference(imgs, infer_tokens)

        meanimg = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        stdimg = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        img_view = input['image'] * stdimg + meanimg
        img_view = img_view.permute(0, 2, 3, 1)
        img_view = (img_view.numpy() * 255).astype(np.uint8)
        result_view = np.zeros([input['image'].shape[0], 256, 256, 3], dtype=np.uint8)

        for i in range(input['image'].shape[0]):

            result_i = result[i]
            result_i = torch.argmax(result_i, dim=0, keepdim=True)
            result_i = result_i.permute(1, 2, 0)
            result_i = result_i.cpu().int().numpy()

            # generate text label
            unique_set = np.unique(result_i)
            result_i_text = ''
            for id in unique_set:
                if id != 0:
                    result_i_text += CLASS_TEXTS[id - 1]
                    if id != unique_set[-1]:
                        result_i_text += ','

            # index to RGB
            result_i_rgb = COLOR_LIST[result_i[:, :, 0]]

            # put text
            cv2.putText(result_i_rgb, result_i_text, org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                        color=(255, 255, 255), thickness=2)

            result_view[i] = result_i_rgb

        img_view = make_img_gird(img_view, 8)
        result_view = make_img_gird(result_view, 8)

        cv2.imwrite(f'visualization/{str(epoch)}_{str(idx)}_img.png', img_view)
        cv2.imwrite(f'visualization/{str(epoch)}_{str(idx)}_out.png', result_view)

        self.net.train()


def main(rank, world_size, cfg, dataset):
    print('rank: ', rank)
    ddp_setup(rank=rank, world_size=world_size)

    sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=4,
                        pin_memory=True, sampler=sampler, collate_fn=collate_fn)

    cfg['train']['num_instances'] = len(loader) * cfg.train.batch_size

    trainer = Trainer(cfg, rank)
    trainer.build_scheduler(len(loader))
    distilbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',
                                                         TOKENIZERS_PARALLELISM=False)

    epoch = 0
    max_epoch = cfg.train.epochs
    loss_meter = 0
    # trainer.resume(epoch)
    while epoch < max_epoch:
        loader.sampler.set_epoch(epoch)
        num_steps = len(loader)
        for idx, samples in enumerate(loader):
            all_images = samples['image'].cuda()
            all_texts = distilbert_tokenizer(samples['raw_caption'], return_tensors='pt', padding='max_length',
                                             truncation=True, max_length=77)
            all_crossimage = samples['cross_image'].cuda()
            cross_entity = distilbert_tokenizer(samples['cross_entity'], return_tensors='pt', padding='max_length',
                                                truncation=True, max_length=77)
            cross_entity = {key: val.cuda() for key, val in cross_entity.items()}

            loss_dict = trainer.update(image=all_images, text=all_texts, cross_image=all_crossimage,
                                       cross_entity=cross_entity)
            trainer.lr_scheduler.step_update(epoch * num_steps + idx)
            loss_meter += loss_dict['loss'].item()

            if idx % 200 == 0:
                print('[%d/%d %d/%d] loss: %.4f matching_loss: %.4f mask_loss: %.4f'
                      % (idx, num_steps, epoch, max_epoch, loss_meter / 200, loss_dict['matching'].item(),
                         loss_dict['mask'].item()))
                loss_meter = 0
                if dist.get_rank() == 0:
                    trainer.show_result(samples, distilbert_tokenizer, epoch, idx)

        trainer.save()

        epoch += 1


if __name__ == '__main__':
    cfg = OmegaConf.load(r'configs/config.yaml')
    world_size = torch.cuda.device_count()
    print('world_size: ', world_size)

    linear_scaled_lr = cfg.train.base_lr * cfg.train.batch_size * world_size / 4096.0
    linear_scaled_warmup_lr = cfg.train.warmup_lr * cfg.train.batch_size * world_size / 4096.0
    linear_scaled_min_lr = cfg.train.min_lr * cfg.train.batch_size * world_size / 4096.0
    cfg.train.base_lr = linear_scaled_lr
    cfg.train.warmup_lr = linear_scaled_warmup_lr
    cfg.train.min_lr = linear_scaled_min_lr

    dataset = Dataset_GID_train(cfg)
    mp.spawn(main, args=(world_size, cfg, dataset), nprocs=world_size)

