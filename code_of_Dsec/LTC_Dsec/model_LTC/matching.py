# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn
import torch as th

from einops import rearrange
from model_LTC import network_blocks
# from model_LTC.deform import DeformConv2d,DeformBottleneck,DeformSimpleBottleneck


def _pad_zero_columns_from_left(tensor, number_of_columns):
    return nn.ZeroPad2d((number_of_columns, 0, 0, 0))(tensor)


class Matching(nn.Module):
    def __init__(self, maximum_disparity, operation):
        """Returns initialized matching module.

        Args:
            maximum_disparity: Upper limit of disparity range
                               [0, maximum_disparity].
            operation: Operation that is applied to concatenated
                       left-right discriptors for all disparities.
                       This can be network module of function.
        """
        super(Matching, self).__init__()
        self._maximum_disparity = maximum_disparity
        self._operation = operation

    def set_maximum_disparity(self, maximum_disparity):
        self._maximum_disparity = maximum_disparity

    def forward(self, left_embedding, right_embedding):
        """Returns concatenated compact matching signatures for every disparity.

        Args:
            left_embedding, right_embedding: Tensors for left and right
                            image embeddings with indices
                            [batch_index, feature_index, y, x].

        Returns:
            matching_signature: 4D tensor that contains concatenated
                                matching signatures (or matching score,
                                depending on "operation") for every disparity.
                                Tensor has indices
                                [batch_index, feature_index, disparity_index,
                                 y, x].
        """
        # padded_right_embedding = _pad_zero_columns_from_left(
        #     right_embedding, self._maximum_disparity)
        # matching_signatures = []
        # concatenated_embedding = th.cat([left_embedding, right_embedding],
        #                                 dim=1)
        # matching_signatures.append(self._operation(concatenated_embedding))
        # for disparity in range(1, self._maximum_disparity + 1):
        #     shifted_right_embedding = \
        #         padded_right_embedding[:, :, :,
        #          self._maximum_disparity - disparity:-disparity]
        #     concatenated_embedding = th.cat(
        #         [left_embedding, shifted_right_embedding], dim=1)
        #     matching_signatures.append(self._operation(concatenated_embedding))
        # # [B, 8, 16, 80, 96]
        # # print("maximum_disparity, len, matching_signatures",self._maximum_disparity,len(matching_signatures),th.stack(matching_signatures,dim=2).size())
        # return th.stack(matching_signatures, dim=2)
       
        padded_right_embedding = _pad_zero_columns_from_left(
            right_embedding, self._maximum_disparity)
        matching_signatures = []
        concatenated_embedding = th.cat([left_embedding, right_embedding],dim=1)
        # matching_signatures.append(self._operation(concatenated_embedding))
        matching_signatures.append(concatenated_embedding)
        for disparity in range(1, self._maximum_disparity + 1):
            shifted_right_embedding = \
                padded_right_embedding[:, :, :,
                 self._maximum_disparity - disparity:-disparity]
            concatenated_embedding = th.cat(
                [left_embedding, shifted_right_embedding], dim=1)
            matching_signatures.append(concatenated_embedding)

        matching_signatures_5d = th.stack(matching_signatures, dim=2)
        matching_signatures_5d = rearrange(matching_signatures_5d, 'b c d h w -> (b d) c h w')
        
        after_operation = rearrange(self._operation(matching_signatures_5d), '(b d) c h w -> b c d h w', d = self._maximum_disparity+1)
        return after_operation 
        # return matching_signatures_5d
        # matching_signatures.append(self._operation(concatenated_embedding))
        # print("maximum_disparity, len, matching_signatures",self._maximum_disparity,len(matching_signatures),th.stack(matching_signatures,dim=2).size())
        # return th.stack(matching_signatures, dim=2)


class MatchingOperation(nn.Module):
    """Operation applied to concatenated left / right descriptors."""

    def __init__(self,
                 number_of_concatenated_descriptor_features=128,
                 number_of_features=64,
                 number_of_compact_matching_signature_features=8,
                 number_of_residual_blocks=2):
        """Returns initialized match operation network.

        For every disparity, left image descriptor is concatenated
        along the feature dimension with shifted by the disparity value
        right image descriptor and passed throught the network.
        """
        super(MatchingOperation, self).__init__()
        matching_operation_modules = [
            network_blocks.convolution_3x3(
                number_of_concatenated_descriptor_features, number_of_features)
            # DeformConv2d(in_channels=number_of_concatenated_descriptor_features, out_channels=number_of_features, kernel_size=3, stride=1, dilation=1, padding=1)
        ]
        matching_operation_modules += [
            network_blocks.ResidualBlock(number_of_features)
            # DeformSimpleBottleneck(number_of_features, number_of_features,mdconv_dilation=1,padding=1)
            for _ in range(number_of_residual_blocks)
        ]
        matching_operation_modules += [
            # DeformConv2d(in_channels=number_of_features, out_channels=number_of_compact_matching_signature_features, kernel_size=3, stride=1, dilation=1, padding=1)
            network_blocks.convolution_3x3(
                number_of_features,
                number_of_compact_matching_signature_features)
        ]
        # self.SEblock = network_blocks.SELayer()
        # matching_operation_modules += [network_blocks.SELayer(number_of_compact_matching_signature_features)]
        # matching_operation_modules += [network_blocks.eca_block(number_of_compact_matching_signature_features)]
        self._matching_operation_modules = nn.ModuleList(
            matching_operation_modules)

    def forward(self, concatenated_descriptors):
        """Returns compact matching signature.

        Args:
            concatenated_descriptors: concatenated left / right image
                                descriptors of size
                                batch_size x 128 x (height / 4) x (width / 4).

        Returns:
            compact_matching_signature: tensor of size
                                batch_size x 8 x (height / 4) x (width / 4).
        """
        compact_matching_signature = concatenated_descriptors
        for _module in self._matching_operation_modules:
            compact_matching_signature = _module(compact_matching_signature)
        return compact_matching_signature
