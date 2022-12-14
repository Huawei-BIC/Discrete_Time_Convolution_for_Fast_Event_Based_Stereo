# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
from torch import nn
from model_LTC import embedding
from model_LTC import estimator
from model_LTC import matching
from model_LTC import regularization
from model_LTC import size_adapter

import time


class PdsNetwork(nn.Module):
    """Practical Deep Stereo (PDS) network."""

    def __init__(self, size_adapter_module, embedding_module, matching_module,
                 regularization_module, estimator_module):
        super(PdsNetwork, self).__init__()
        self._size_adapter = size_adapter_module
        self._embedding = embedding_module
        self._matching = matching_module
        self._regularization = regularization_module
        self._estimator = estimator_module

    def set_maximum_disparity(self, maximum_disparity):
        """Reconfigure network for different disparity range."""
        if (maximum_disparity + 1) % 64 != 0:
            raise ValueError(
                '"maximum_disparity" + 1 should be multiple of 64, e.g.,'
                '"maximum disparity" can be equal to 63, 191, 255, 319...')
        self._maximum_disparity = maximum_disparity
        # During the embedding spatial dimensions of an input are downsampled
        # 4x times. Therefore, "maximum_disparity" of matching module is
        # computed as (maximum_disparity + 1) / 4 - 1.
        self._matching.set_maximum_disparity((maximum_disparity + 1) // 4 - 1)

    def pass_through_network(self, left_image, right_image):
        start_time = time.time()
        left_descriptor, shortcut_from_left = self._embedding(left_image)
        right_descriptor = self._embedding(right_image)[0]
        print("Embedding Duration:{:.4f}s".format(time.time()-start_time))
        print("leftdisc",left_descriptor.size())#  1 32 80 96
        print("sfl",shortcut_from_left.size()) #  1 8 80 96
        start_time = time.time()
        matching_signatures = self._matching(left_descriptor, right_descriptor)
        print("Matching Duration:{:.4f}s".format(time.time()-start_time))
        # print("after matching", matching_signatures.size())
        # after matching : 1 8 16 320 384

        start_time = time.time()
        output = self._regularization(matching_signatures, shortcut_from_left)
        print("Cost volumn:{:.4f}s".format(time.time()-start_time))
        return output, shortcut_from_left


    def forward(self, left_image, right_image):
        """Returns sub-pixel disparity (or matching cost in training mode)."""
        network_output = self.pass_through_network(
            self._size_adapter.pad(left_image),
            self._size_adapter.pad(right_image))[0]
        if not self.training:
            network_output = self._estimator(network_output)
        return self._size_adapter.unpad(network_output)

    @staticmethod
    def default(maximum_disparity=255):
        """Returns network with default parameters."""
        network = PdsNetwork(
            size_adapter_module=size_adapter.SizeAdapter(),
            embedding_module=embedding.Embedding(),
            matching_module=matching.Matching(
                operation=matching.MatchingOperation(), maximum_disparity=0),
            regularization_module=regularization.Regularization(),
            estimator_module=estimator.SubpixelMap())
        network.set_maximum_disparity(maximum_disparity)
        return network
