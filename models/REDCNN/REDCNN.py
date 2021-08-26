from torch.nn import functional as F
import torch
import argparse

from models.basic_template import TrainTask
from .REDCNN_wrapper import Generator

'''
python -m torch.distributed.launch --nproc_per_node=1 --master_port=17677 main.py --batch_size 256 \
    --max_iter 20000 --save_freq 2000 --train_dataset_name cmayo_train_64 --test_dataset_name cmayo_test_512 \
    --hu_min -300 --hu_max 300 --weight_decay 0 \
    --model_name REDCNN --run_name official --num_layers 10 --num_channels 32 --init_lr 1e-4 --min_lr 1e-5
'''


class REDCNN(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--num_layers", default=10, type=int)
        parser.add_argument("--num_channels", default=32, type=int)
        parser.add_argument("--init_lr", default=1e-4, type=float)
        parser.add_argument("--min_lr", default=1e-5, type=float)
        return parser

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = opt.init_lr - n_iter * (opt.init_lr - opt.min_lr) / opt.max_iter

    def set_model(self):
        opt = self.opt
        generator = Generator(in_channels=1, out_channels=opt.num_channels, num_layers=opt.num_layers)
        optimizer = torch.optim.Adam(generator.parameters(), opt.init_lr)

        self.logger.modules = [generator, optimizer]

        self.generator = generator.cuda()
        self.optimizer = optimizer

    def train(self, inputs, n_iter):
        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
        gen_full_dose = self.generator(low_dose)
        loss = F.mse_loss(gen_full_dose, full_dose)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        lr = self.optimizer.param_groups[0]['lr']
        self.logger.msg([loss, lr], n_iter)
