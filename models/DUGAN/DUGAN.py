from torch.nn import functional as F
import torch
import numpy as np
import copy
import torchvision
import argparse
import tqdm
import torch.nn as nn

from models.basic_template import TrainTask
from utils.grad_loss import SobelOperator
from .DUGAN_wrapper import UNet
from models.REDCNN.REDCNN_wrapper import Generator
from utils.gan_loss import ls_gan
from utils.ops import turn_on_spectral_norm
from utils.metrics import compute_ssim, compute_psnr, compute_rmse

'''
python main.py --batch_size=64 --cr_loss_weight=5.08720932695335 --cutmix_prob=0.7615524094697519 --cutmix_warmup_iter=1000 --d_lr=7.122979672016055e-05 --g_lr=0.00018083340390609657 --grad_gen_loss_weight=0.11960717521104237 --grad_loss_weight=35.310016043755894 --img_gen_loss_weight=0.14178356036938378 --max_iter=50000 --model_name=UnetGAN --num_channels=32 --num_layers=10 --num_workers=32 --pix_loss_weight=5.034293425614828 --print_freq=10 --run_name=newest --save_freq=2500 --test_batch_size=8 --test_dataset_name=lmayo_test_512 --train_dataset_name=lmayo_train_64 --use_grad_discriminator=true --weight_decay 0. --num_workers 4
python main.py --batch_size=64 --cr_loss_weight=5.08720932695335 --cutmix_prob=0.7615524094697519 --cutmix_warmup_iter=1000 --d_lr=7.122979672016055e-05 --g_lr=0.00018083340390609657 --grad_gen_loss_weight=0.11960717521104237 --grad_loss_weight=35.310016043755894 --img_gen_loss_weight=0.14178356036938378 --max_iter=100000 --model_name=UnetGAN --num_channels=32 --num_layers=10 --num_workers=32 --pix_loss_weight=5.034293425614828 --print_freq=10 --run_name=newest --save_freq=2500 --test_batch_size=8 --test_dataset_name=cmayo_test_512 --train_dataset_name=cmayo_train_64 --use_grad_discriminator=true --weight_decay 0. --num_workers 4
'''


class DUGAN(TrainTask):

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--num_layers", default=10, type=int)
        parser.add_argument("--num_channels", default=32, type=int)
        # Need D conv_dim 64
        parser.add_argument("--g_lr", default=1e-4, type=float)
        parser.add_argument("--d_lr", default=1e-4, type=float)
        parser.add_argument("--d_iter", default=1, type=int)
        parser.add_argument("--cutmix_prob", default=0.5, type=float)
        parser.add_argument("--img_gen_loss_weight", default=0.1, type=float)
        parser.add_argument("--grad_gen_loss_weight", default=0.1, type=float)
        parser.add_argument("--pix_loss_weight", default=1., type=float)
        parser.add_argument("--grad_loss_weight", default=20., type=float)
        parser.add_argument("--cr_loss_weight", default=1.0, type=float)
        parser.add_argument("--cutmix_warmup_iter", default=1000, type=int)
        parser.add_argument("--use_grad_discriminator", help='use_grad_discriminator', type=bool, default=True)
        parser.add_argument("--moving_average", default=0.999, type=float)
        parser.add_argument("--repeat_num", default=6, type=int)
        return parser

    def set_model(self):
        opt = self.opt
        generator = Generator(in_channels=1, out_channels=opt.num_channels, num_layers=opt.num_layers, kernel_size=3,
                              padding=1)
        g_optimizer = torch.optim.Adam(generator.parameters(), opt.g_lr, weight_decay=opt.weight_decay)

        self.gan_metric = ls_gan
        img_discriminator = UNet(repeat_num=opt.repeat_num, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        img_discriminator = turn_on_spectral_norm(img_discriminator)
        img_d_optimizer = torch.optim.Adam(img_discriminator.parameters(), opt.d_lr)
        grad_discriminator = copy.deepcopy(img_discriminator)
        grad_d_optimizer = torch.optim.Adam(grad_discriminator.parameters(), opt.d_lr)

        ema_generator = copy.deepcopy(generator)

        self.logger.modules = [generator, g_optimizer, img_discriminator, img_d_optimizer, grad_discriminator,
                               grad_d_optimizer, ema_generator]

        self.sobel = SobelOperator().cuda()
        self.generator = generator.cuda()
        self.g_optimizer = g_optimizer
        self.img_discriminator = img_discriminator.cuda()
        self.img_d_optimizer = img_d_optimizer
        self.grad_discriminator = grad_discriminator.cuda()

        self.ema_generator = ema_generator.cuda()

        self.grad_d_optimizer = grad_d_optimizer
        self.apply_cutmix_prob = torch.rand(opt.max_iter)

    def train_discriminator(self, discriminator, d_optimizer,
                            full_dose, low_dose, gen_full_dose, prefix, n_iter=0):
        opt = self.opt
        msg_dict = {}
        ############## Train Discriminator ###################
        d_optimizer.zero_grad()
        real_enc, real_dec = discriminator(full_dose)
        fake_enc, fake_dec = discriminator(gen_full_dose.detach())
        source_enc, source_dec = discriminator(low_dose)
        msg_dict.update({
            'enc/{}_real'.format(prefix): real_enc,
            'enc/{}_fake'.format(prefix): fake_enc,
            'enc/{}_source'.format(prefix): source_enc,
            'dec/{}_real'.format(prefix): real_dec,
            'dec/{}_fake'.format(prefix): fake_dec,
            'dec/{}_source'.format(prefix): source_dec,
        })

        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                    self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                    self.gan_metric(source_enc, 0.) + self.gan_metric(source_dec, 0.)
        total_loss = disc_loss

        apply_cutmix = self.apply_cutmix_prob[n_iter - 1] < warmup(opt.cutmix_warmup_iter, opt.cutmix_prob, n_iter)
        if apply_cutmix:
            mask = cutmix(real_dec.size()).to(real_dec)

            # if random.random() > 0.5:
            #     mask = 1 - mask

            cutmix_enc, cutmix_dec = discriminator(mask_src_tgt(full_dose, gen_full_dose.detach(), mask))

            cutmix_disc_loss = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)

            cr_loss = F.mse_loss(cutmix_dec, mask_src_tgt(real_dec, fake_dec, mask))

            total_loss += cutmix_disc_loss + cr_loss * opt.cr_loss_weight

            msg_dict.update({
                'enc/{}_cutmix'.format(prefix): cutmix_enc,
                'dec/{}_cutmix'.format(prefix): cutmix_dec,
                'loss/{}_cutmix_disc'.format(prefix): cutmix_disc_loss,
                'loss/{}_cr'.format(prefix): cr_loss,
            })

        total_loss.backward()

        d_optimizer.step()
        self.logger.msg(msg_dict, n_iter)

    def update_moving_average(self):
        opt = self.opt
        m = opt.moving_average
        for old_param, new_param in zip(self.ema_generator.parameters(), self.generator.parameters()):
            old_param.data = old_param.data * m + new_param.data * (1. - m)

    def train(self, inputs, n_iter):
        opt = self.opt

        self.update_moving_average()

        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

        self.generator.train()
        self.img_discriminator.train()
        self.grad_discriminator.train()
        msg_dict = {}

        gen_full_dose = self.generator(low_dose)
        grad_gen_full_dose = self.sobel(gen_full_dose)
        grad_low_dose = self.sobel(low_dose)
        grad_full_dose = self.sobel(full_dose)
        self.train_discriminator(self.img_discriminator, self.img_d_optimizer,
                                 full_dose, low_dose, gen_full_dose, prefix='img', n_iter=n_iter)

        if n_iter % opt.d_iter == 0:
            ############## Train Generator ###################

            ########### GAN Loss ############
            self.g_optimizer.zero_grad()
            img_gen_enc, img_gen_dec = self.img_discriminator(gen_full_dose)
            img_gen_loss = self.gan_metric(img_gen_enc, 1.) + self.gan_metric(img_gen_dec, 1.)

            total_loss = 0.
            if opt.use_grad_discriminator:
                self.train_discriminator(self.grad_discriminator, self.grad_d_optimizer,
                                         grad_full_dose, grad_low_dose, grad_gen_full_dose, prefix='grad',
                                         n_iter=n_iter)
                grad_gen_enc, grad_gen_dec = self.grad_discriminator(grad_gen_full_dose)
                grad_gen_loss = self.gan_metric(grad_gen_enc, 1.) + self.gan_metric(grad_gen_dec, 1.)
                total_loss = grad_gen_loss * opt.grad_gen_loss_weight
                msg_dict.update({
                    'enc/grad_gen_enc': grad_gen_enc,
                    'dec/grad_gen_dec': grad_gen_dec,
                    'loss/grad_gen_loss': grad_gen_loss,
                })

            ########### Pixel Loss ############
            pix_loss = F.mse_loss(gen_full_dose, full_dose)

            ########### L1 Loss ############
            l1_loss = F.l1_loss(gen_full_dose, full_dose)

            ########### Grad Loss ############
            grad_loss = F.l1_loss(grad_gen_full_dose, grad_full_dose)

            total_loss += img_gen_loss * opt.img_gen_loss_weight + \
                          pix_loss * opt.pix_loss_weight + \
                          grad_loss * opt.grad_loss_weight

            total_loss.backward()

            self.g_optimizer.step()
            msg_dict.update({
                'enc/img_gen_enc': img_gen_enc,
                'dec/img_gen_dec': img_gen_dec,
                'loss/img_gen_loss': img_gen_loss,
                'loss/pix': pix_loss,
                'loss/l1': l1_loss,
                'loss/grad': grad_loss,
            })
            self.logger.msg(msg_dict, n_iter)

    @torch.no_grad()
    def generate_images(self, n_iter):
        self.generator.eval()
        low_dose, full_dose = self.test_images
        bs, ch, w, h = low_dose.size()
        gen_full_dose = self.generator(low_dose).clamp(0., 1.)
        fake_imgs = [low_dose, full_dose, gen_full_dose,
                     self.img_discriminator(gen_full_dose)[1].clamp(0., 1.),
                     self.grad_discriminator(self.sobel(gen_full_dose))[1].clamp(0., 1.)]
        fake_imgs = torch.stack(fake_imgs).transpose(1, 0).reshape((-1, ch, w, h))
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=5), n_iter, 'test')

    @torch.no_grad()
    def test(self, n_iter):
        self.generator.eval()
        self.ema_generator.eval()
        for name, generator in zip(['ema_', ''], [self.ema_generator, self.generator]):
            psnr_score, ssim_score, rmse_score, total_num = 0., 0., 0., 0
            for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
                batch_size = low_dose.size(0)
                low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
                gen_full_dose = generator(low_dose).clamp(0., 1.)
                psnr_score += compute_psnr(gen_full_dose, full_dose) * batch_size
                ssim_score += compute_ssim(gen_full_dose, full_dose) * batch_size
                rmse_score += compute_rmse(gen_full_dose, full_dose) * batch_size
                total_num += batch_size
            psnr = psnr_score / total_num
            ssim = ssim_score / total_num
            rmse = rmse_score / total_num

            self.logger.msg({'{}ssim'.format(name): ssim,
                             '{}psnr'.format(name): psnr,
                             '{}rmse'.format(name): rmse}, n_iter)


def warmup(warmup_iter, cutmix_prob, n_iter):
    return min(n_iter * cutmix_prob / warmup_iter, cutmix_prob)


def cutmix(mask_size):
    mask = torch.ones(mask_size)
    lam = np.random.beta(1., 1.)
    _, _, height, width = mask_size
    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))
    mask[:, :, y0:y1, x0:x1] = 0
    return mask


def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target
