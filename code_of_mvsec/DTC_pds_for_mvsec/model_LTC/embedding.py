# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn

from model_LTC import network_blocks
# from model_LTC.deform import DeformConv2d,DeformBottleneck,DeformSimpleBottleneck


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
# 
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)



class Embedding(nn.Module):
    """Embedding module."""

    def __init__(self,
                 number_of_input_features=3,
                 number_of_embedding_features=16,
                 number_of_shortcut_features=8,
                 number_of_residual_blocks=2):
        """Returns initialized embedding module.

        Args:
            number_of_input_features: number of channels in the input image;
            number_of_embedding_features: number of channels in image's
                                          descriptor;
            number_of_shortcut_features: number of channels in the redirect
                                         connection descriptor;
            number_of_residual_blocks: number of residual blocks in embedding
                                       network.
        """
        super(Embedding, self).__init__()
        embedding_modules = [
            nn.InstanceNorm2d(number_of_input_features),
            network_blocks.convolutional_block_5x5_stride_2(
                number_of_input_features, number_of_embedding_features),
            network_blocks.convolutional_block_5x5_stride_2(
                number_of_embedding_features, number_of_embedding_features),
            # DeformConv2d(in_channels=number_of_input_features, out_channels=number_of_embedding_features, kernel_size=5, stride=2, dilation=1, padding=2),
            # DeformConv2d(in_channels=number_of_embedding_features, out_channels=number_of_embedding_features, kernel_size=5, stride=2, dilation=1, padding=2)
        ]
        embedding_modules += [
            network_blocks.ResidualBlock(number_of_embedding_features)
            # DeformSimpleBottleneck(number_of_embedding_features, number_of_embedding_features,mdconv_dilation=1,padding=1)
            for _ in range(number_of_residual_blocks)
        ]
        
        ###################################channel-sise attention##############################
        # embedding_modules += [network_blocks.eca_block(number_of_embedding_features)]
        # embedding_modules += [network_blocks.SELayer(number_of_embedding_features)]
        
        #########################################################################################
        
        
        
        self._embedding_modules = nn.ModuleList(embedding_modules)
        self._shortcut = network_blocks.convolutional_block_3x3(
            number_of_embedding_features, number_of_shortcut_features)
        # self.shortcut_eca_block = network_blocks.eca_block(number_of_shortcut_features)
        # self._shortcut = DeformConv2d(in_channels=number_of_embedding_features, out_channels=number_of_shortcut_features, kernel_size=3, stride=1, dilation=1, padding=1)


    def forward(self, image):
        """Returns image's descriptor and redirect connection descriptor.

        Args:
            image: color image of size
                   batch_size x 3 x height x width;

        Returns:
            descriptor: image's descriptor of size
                        batch_size x 64 x (height / 4) x (width / 4);
            shortcut_from_left_image: shortcut connection from left image
                      descriptor (it is used in regularization network). It
                      is tensor of size
                      (batch_size, 8, height / 4, width / 4).
        """
        descriptor = image
        for embedding_module in self._embedding_modules:
            descriptor = embedding_module(descriptor)
        # short_cut = self.shortcut_eca_block(self._shortcut(descriptor))
        return descriptor, self._shortcut(descriptor)
        # return descriptor, short_cut
