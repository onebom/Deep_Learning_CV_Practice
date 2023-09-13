import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding

import numpy as np

class Discriminator(nn.Module):
    
    def __init__(self, resolution, num_channels=3, conditional=False,
                 n_classes=0, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4,
                 mbstd_num_features=1, blur_filter=None, structure='linear',
                 **kwargs):
        """
        Discriminator used in the StyleGAN paper.

        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
        # label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param mbstd_group_size: Group size for the mini_batch standard deviation layer, 0 = disable.
        :param mbstd_num_features: Number of features for the mini_batch standard deviation layer.
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """
        super(Discriminator, self).__init__()

        if conditional:
            assert n_classes > 0, "Conditional Discriminator requires n_class > 0"
            # self.embedding = nn.Embedding(n_classes, num_channels * resolution ** 2)
            num_channels *= 2
            self.embeddings = []

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.conditional = conditional
        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure
        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, use_wscale=use_wscale, activation_layer=act,
                                             blur_kernel=blur_filter))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))
            # Create embeddings for various inputs:
            if conditional:
                r = 2 ** (res)
                self.embeddings.append(
                    Embedding(n_classes, (num_channels // 2) * r * r))

        if self.conditional:
            self.embeddings.append(nn.Embedding(
                n_classes, (num_channels // 2) * 4 * 4))
            self.embeddings = nn.ModuleList(self.embeddings)

        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, images_in, depth, alpha=1., labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """
        
        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.conditional:
            assert labels_in is not None, "Conditional Discriminator requires labels"
        # print(embedding_in.shape, images_in.shape)
        # exit(0)
        # print(self.embeddings)
        # exit(0)
        if self.structure == 'fixed':
            if self.conditional:
                embedding_in = self.embeddings[0](labels_in)
                embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                 images_in.shape[2],
                                                 images_in.shape[3])
                images_in = torch.cat([images_in, embedding_in], dim=1)
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)
            
        elif self.structure == 'linear':
            if depth > 0:
                if self.conditional:
                    embedding_in = self.embeddings[self.depth -
                                                   depth - 1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)
                    
                residual = self.from_rgb[self.depth -
                                         depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth -
                                       1](self.from_rgb[self.depth - depth - 1](images_in))
                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[(self.depth - depth):]:
                    x = block(x)
            else:
                if self.conditional:
                    embedding_in = self.embeddings[-1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)
                x = self.from_rgb[-1](images_in)
                    
            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out