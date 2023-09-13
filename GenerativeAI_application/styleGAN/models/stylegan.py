import os
import copy
import time
import timeit
import datetime

import torch
import torch.nn as nn
import numpy as np

import model.generator as Generator
import model.discriminator as Discriminator
import model.losses as Losses

class styleGAN(object):
    def __init__(self, args):
        """
        wrapper around the generator and the discirimiatior.

        Args:
            structure: 'fixed' = no progressive growing, 'linear' = human-readable
            resolution: Input resolution. Overridden based on dataset.
            device: device to run the GAN on (GPU / CPU)
            d_repeats: How many times the discriminator is trained per G iteration.
            use_ema: boolean for whether to use exponential moving averages
            ema_decay: value of mu for ema

            drift: drift penalty for the (Used only if loss is wgan or wgan-gp)
        """
        assert args.structure in ['fixed','linear']

        if args.conditional:
            assert args.n_classes>0, "Conditaional GANs require n_classes>0"

        self.structure = args.structure
        self.depth=int(np.log2(args.resolution))-1
        self.latent_size=args.latent_size
        self.device = args.device
        self.d_repeats = args.d_repeats
        self.conditional = args.conditional
        self.n_classes = args.n_classes

        self.use_ema = args.use_ema
        self.ema_decay = args.ema_decay

        self.gen = Generator(num_channels=args.num_channels,
                             resolution=args.resolution,
                             structure=self.structure,
                             conditional=self.conditional,
                             n_classes=self.n_classes,
                             **args.g_args).to(self.device)

        self.dis = Discriminator(num_channels=args.num_channels,
                                 resolution=args.resolution,
                                 structure=self.structure,
                                 conditional=self.conditional,
                                 n_classes=self.n_classes,
                                 **args.d_args).to(self.device)

        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=args.g_opt_args.learning_rate, betas=(args.g_opt_args.beta_1, args.g_opt_args.beta_2), eps=args.g_opt_args.eps)
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=args.d_opt_args.learning_rate, betas=(args.d_opt_args.beta_1, args.d_opt_args.beta_2), eps=args.d_opt_args.eps)

        self.drift = args.drift
        self.loss = self.__setup_loss(args.loss)

        # if self.use_ema:
        #     # create a shadow copy of the generator
        #     self.gen_shadow = copy.deepcopy(self.gen)
        #     # updater function:
        #     self.ema_updater = update_average
        #     # initialize the gen_shadow weights equal to the weights of gen
        #     self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_loss(self, loss):
            if isinstance(loss, str):
                loss = loss.lower()  # lowercase the string
                
                if not self.conditional:
                    assert loss in ["logistic", "hinge", "standard-gan",
                                    "relativistic-hinge"], "Unknown loss function"
                    if loss == "logistic":
                        loss_func = Losses.LogisticGAN(self.dis)
                    elif loss == "hinge":
                        loss_func = Losses.HingeGAN(self.dis)
                    if loss == "standard-gan":
                        loss_func = Losses.StandardGAN(self.dis)
                    elif loss == "relativistic-hinge":
                        loss_func = Losses.RelativisticAverageHingeGAN(self.dis)
                else:
                    assert loss in ["conditional-loss"]
                    if loss == "conditional-loss":
                        loss_func = Losses.ConditionalGANLoss(self.dis)

            return loss_func

    def train(self, dataset, num_workers, epochs, batch_sizes, 
    fade_in_percentage, logger, output,
    num_samples=36, start_depth=0, feedback_factor=100, 
    checkpoint_factor=1):

        self.gen.train()
        self.dis.train()
        # if self.use_ema:
            # self.gen_shadow.train()

        global_time=time.time()
        fixed_input=torch.randn(num_samples, self.latent_size).to(self.device)
        fixed_labels=None

        if self.conditional:
            fixed_label = torch.linespace(0, self.n_classes - 1, num_samples).to(torch.int64).to(self.device)

        logger.info("Starting the training process ... \n")
        if self.structure == 'fixed':
            start_depth = self.depth -1
        step=1
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2,current_depth+2)
            logger.info("Currently working on depth: %d", current_depth+1)
            logger.info("Current resolution: %d x %d"%(current_res,current_res ))
        
            ticker =1
            data = get_data_loader(dataset, batch_size[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth]+1):
                start = timeit.default_timer()
                logger.info("Epoch:[%d]"%epoch)
                total_batches = len(data)
                
                fade_point=int((fade_in_percentage[current_depth]/100)*epochs[current_depth]*total_batches)
                for i, batch in enumerate(data,1):
                    alpha = ticker /fade_point if ticker <= fade_point else 1

                    if self.conditional:
                        images, labels = batch
                        labels = labels.to(self.device)

                    else:
                        images = batch
                        labels = None
                    
                    images = images.to(self.deive)

                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha, labels)
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha, labels)

                    if i % int(total_batches / feedback_factor+1)==0 or i ==1:
                        elapsed = time.time()-global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
                            % (elapsed, step, i, dis_loss, gen_loss))

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        with torch.no_grad():                            
                            self.create_grid(
                                samples=self.gen(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach() if not self.use_ema
                                else self.gen_shadow(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach(),
                                scale_factor=int(
                                    np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                                img_file=gen_img_file,
                            )
                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1
                
                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    save_dir = os.path.join(output, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    gen_optim_save_file = os.path.join(
                        save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_optim_save_file = os.path.join(
                        save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")

                    torch.save(self.gen.state_dict(), gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(
                            save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                        logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

        logger.info('Training completed.\n')
                        
    def optimize_generator(self, noise, real_batch, depth, alpha, labels=None):
        real=self.__progressive_down_sampling(real_batch,depth,alpha)
        fake=self.gen(noise,depth,alpha,labels)

        if not self.conditional:
            loss=self.loss.gen_loss(real, fake, depth, alpha)
        else:
            loss=self.loss.gen_loss(real,fake,labels,depth,alpha)
        
        self.gen_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=10.)
        self.gen_optim.step()

        return loss.item


    def optimize_discriminator(self, noise, real_batch, depth, alpha, labels=None):
        real = self.__progressive_down_sampling(real_batch, depth, alpha)
        loss_val = 0

        for _ in range(self.d_repeats):
            fake = self.gen(noise, depth, alpha, labels).detach()
            if not self.conditional:
                loss = self.loss.dis_loss(real, fake, depth, alpha)
            else:
                loss = self.loss.dis_loss(real, fake, labels, depth, alpha)

            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()
        return loss_val / self.d_repeats
        
    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """private helper for down_sampling the original images
        """

        from torch.nn import AvgPool2d
        from torch.nn.functinal import interpolate

        if self.structure == 'fixed':
            return real_batch
        
        down_sample_factor = int(np.power(2,self.depth-depth-1))
        prior_down_sample_factor = max(int(np.power(2, self.depth-depth)),0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)
        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples=ds_real_samples
        
        real_samples = (alpha*ds_real_samples)+((1-alpha)*prior_ds_real_samples)

        return real_samples

    #  def check_cuda(self, cuda_flag=False):
    # def train():
    # def evaluate():
    # def save_model():
    # def load_model():
