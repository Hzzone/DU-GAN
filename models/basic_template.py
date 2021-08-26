import torch.utils.data
import torch
import torchvision
import os.path as osp
import tqdm
import argparse
import torch.distributed as dist

from utils.dataset import dataset_dict
from utils.metrics import compute_ssim, compute_psnr, compute_rmse
from utils.loggerx import LoggerX
from utils.sampler import RandomSampler


class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root=osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', '{}_{}'.format(opt.model_name, opt.run_name)))
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.generator = None
        self.set_loader()
        self.set_model()

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')

        parser.add_argument('--save_freq', type=int, default=1000,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=1,
                            help='test_batch_size')
        parser.add_argument('--num_workers', type=int, default=16,
                            help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=20000,
                            help='number of training epochs')
        parser.add_argument('--resume_iter', type=int, default=0,
                            help='number of training epochs')
        parser.add_argument("--local_rank", default=0, type=int)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum')

        # dataset
        parser.add_argument('--train_dataset_name', type=str, default='cmayo_train_64')
        parser.add_argument('--test_dataset_name', type=str, default='cmayo_test_512')
        parser.add_argument('--hu_min', type=int, default=-300)
        parser.add_argument('--hu_max', type=int, default=300)

        parser.add_argument('--run_name', type=str, default='default', help='each run name')
        parser.add_argument('--model_name', type=str, help='the type of method', default='supcon')

        # learning rate
        parser.add_argument('--learning_rate', type=float, default=0.05,
                            help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=int, nargs='*', default=[700, 800, 900],
                            help='where to decay lr, can be a list')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                            help='decay rate for learning rate')
        parser.add_argument('--warmup_from', type=float, default=0.01,
                            help='the initial learning rate if warmup')
        parser.add_argument('--warmup_epochs', type=int, default=0,
                            help='warmup epochs')
        return parser

    @staticmethod
    def build_options():
        pass

    def set_loader(self):
        opt = self.opt

        train_dataset = dataset_dict[opt.train_dataset_name](hu_range=(opt.hu_min, opt.hu_max), transforms=None)
        train_sampler = RandomSampler(train_dataset, batch_size=opt.batch_size,
                                      num_iter=opt.max_iter,
                                      restore_iter=opt.resume_iter)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

        test_dataset = dataset_dict[opt.test_dataset_name](hu_range=(opt.hu_min, opt.hu_max))
        test_images = [test_dataset[i] for i in range(0, min(300, len(test_dataset)), 50)]
        low_dose = torch.stack([x[0] for x in test_images], dim=0).cuda()
        full_dose = torch.stack([x[1] for x in test_images], dim=0).cuda()
        self.test_images = (low_dose, full_dose)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                                  num_workers=opt.num_workers, pin_memory=True)
        self.test_loader = test_loader
        self.train_loader = train_loader

    def fit(self):
        opt = self.opt
        if opt.resume_iter > 0:
            self.logger.load_checkpoints(opt.resume_iter)
        # training routine
        loader = iter(self.train_loader)
        for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
            inputs = next(loader)
            self.adjust_learning_rate(n_iter)
            self.train(inputs, n_iter)
            if n_iter % opt.save_freq == 0:
                self.logger.checkpoints(n_iter)
                self.test(n_iter)
                self.generate_images(n_iter)

    def set_model(opt):
        pass

    def train(self, inputs, n_iter):
        pass

    @torch.no_grad()
    def test(self, n_iter):
        self.generator.eval()
        psnr_score, ssim_score, rmse_score, total_num = 0., 0., 0., 0
        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            batch_size = low_dose.size(0)
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
            gen_full_dose = self.generator(low_dose).clamp(0., 1.)
            psnr_score += compute_psnr(gen_full_dose, full_dose) * batch_size
            ssim_score += compute_ssim(gen_full_dose, full_dose) * batch_size
            rmse_score += compute_rmse(gen_full_dose, full_dose) * batch_size
            total_num += batch_size
        psnr = psnr_score / total_num
        ssim = ssim_score / total_num
        rmse = rmse_score / total_num
        self.logger.msg([psnr, ssim, rmse], n_iter)

    def adjust_learning_rate(self, n_iter):
        opt = self.opt

        pass

    @torch.no_grad()
    def generate_images(self, n_iter):
        self.generator.eval()
        low_dose, full_dose = self.test_images
        bs, ch, w, h = low_dose.size()
        fake_imgs = [low_dose, full_dose, self.generator(low_dose).clamp(0., 1.)]
        fake_imgs = torch.stack(fake_imgs).transpose(1, 0).reshape((-1, ch, w, h))
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=3), n_iter, 'test')
