# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import os
import time

import torch as th
# import torch.distributed as dist
from model_LTC import visualization
from model_LTC import dataset
from torch.utils import data

# import trainer
import numpy as np

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _is_on_cuda(network):
    return next(network.parameters()).is_cuda


def _is_logging_required(example_index, number_of_examples):
    """Returns True only if logging is required.

    Logging is performed after 10%, 20%, ... 100% percents of examples
    is processed.
    """
    return (example_index + 1) % max(1, number_of_examples // 10) == 0


def _set_fastest_cuda_mode():
    th.backends.cudnn.fastest = True
    th.backends.cudnn.benchmark = True
    # pass

def _move_tensors_to_cuda(dictionary_of_tensors):
    if isinstance(dictionary_of_tensors, dict):
        return {
            key: _move_tensors_to_cuda(value)
            for key, value in dictionary_of_tensors.items()
        }
    # return dictionary_of_tensors.cuda(th.device("cuda",dist.get_rank()))
    return dictionary_of_tensors.cuda()

class Trainer(object):
    def __init__(self, parameters):
        """Returns initialized trainer object.

        Args:
            parameters: dictionary with parameters, that
                        should have the same names as
                        attributes of the class (but without
                        underscore).
        """
        if th.cuda.is_available():
            _set_fastest_cuda_mode()
        self._experiment_folder = None
        self._spec_title = None
        self._current_epoch = 0
        self._end_epoch = None
        self._learning_rate_scheduler = None
        self._network = None
        self._training_set_loader = None
        self._test_set_loader = None
        self._optimizer = None
        self._criterion = None
        self._current_losses = []
        self._current_errors = []
        self._current_processing_times = []
        self._training_losses = []
        self._test_errors = []
        self._number_of_examples_to_visualize = 2000 
        self.from_dictionary(parameters)

    def from_dictionary(self, parameters):
        attributes = vars(self)
        for key, value in parameters.items():
            _key = '_{0}'.format(key)
            attributes[_key] = value

    def _initialize_filenames(self):
        self._log_filename = os.path.join(self._experiment_folder, 'log.txt')
        self._plot_filename = os.path.join(self._experiment_folder, 'plot.png')
        self._checkpoint_template = os.path.join(self._experiment_folder,
                                                 '{0:03d}_checkpoint.bin')

    def load_checkpoint(self, filename, load_only_network=False):
        """Initilizes trainer from checkpoint.

        Args:
            filename: file with the checkpoint.
            load_only_network: if the flag is set, the function only loads
                               the network (can be usefull for
                               fine-tuning).
        """
        checkpoint = th.load(filename)
        self._network.load_state_dict(checkpoint['network'])
        if load_only_network:
            return
        parameters = {
            'current_epoch': len(checkpoint['training_losses']),
            'training_losses': checkpoint['training_losses'],
            'test_errors': checkpoint['test_errors']
        }
        self.from_dictionary(parameters)
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._learning_rate_scheduler.load_state_dict(
            checkpoint['learning_rate_scheduler'])

    def _save_checkpoint(self):
        th.save({
            'training_losses':
            self._training_losses,
            'test_errors':
            self._test_errors,
            'network':
            self._network.state_dict(),
            'optimizer':
            self._optimizer.state_dict(),
            'learning_rate_scheduler':
            self._learning_rate_scheduler.state_dict()
        }, self._checkpoint_template.format(self._current_epoch + 1))

    def train(self):
        """Trains network and returns validation error of last epoch."""
        self._initialize_filenames()
        self._logger = visualization.Logger(self._log_filename)
        start_epoch = self._current_epoch
        if start_epoch == self._end_epoch:
            return None
        self._logger.log("Training started.")
        for self._current_epoch in range(start_epoch, self._end_epoch):
            self._training_losses.append(self._train_for_epoch(self._current_epoch))
            self._test_errors.append(self._test()[0])
            self._report_training_progress()
            if type(self._learning_rate_scheduler) == tuple:
                self._learning_rate_scheduler[0].step()
                self._learning_rate_scheduler[2].step()
            else:
                self._learning_rate_scheduler.step()
            self._save_checkpoint()
        self._current_epoch = self._end_epoch
        # self.save_dataset_and_CFC()
        return self._test_errors[-1]

    def _run_network_and_measure_time(self, example):
        if th.cuda.is_available():
            th.cuda.synchronize()
        start_time = time.time()
        self._run_network(example)
        end_time = time.time()
        if th.cuda.is_available():
            th.cuda.synchronize()
        example['processing_time'] = float(time.time() - start_time)
        print("Total Duration:{:.4f}s".format(end_time-start_time))
        print("FPS:{:.4f}".format(1/(end_time-start_time)))


    def _report_test_results(self, error, processing_time):
        """Reports test results."""
        raise NotImplementedError('"_report_test_results" method should '
                                  'be implemented in a child class.')

    def _run_network(self, batch_or_example):
        """Runs network and adds output to "batch_or_example"."""
        raise NotImplementedError('"_run_network" method should '
                                  'be implemented in a child class.')

    def _compute_gradients_wrt_loss(self, batch):
        """Computes loss, gradients w.r.t loss and saves loss.

        The loss should be saved to "loss" item of "batch".
        """
        raise NotImplementedError('"_compute_loss" method should '
                                  'be implemented in a child class.')

    def _compute_error(self, example):
        """Computes error and adds it to "example" as an "error" item."""
        raise NotImplementedError('"_compute_error" method should '
                                  'be implemented in a child class.')

    def _visualize_example(self, example, example_index):
        """Visualize result for the example during validation and test.

        Args:
            example: should include network input and output necessary for
                     the visualization.
            example_index: index of the example.
        """
        raise NotImplementedError('"_visualize_example" method should '
                                  'be implemented in a child class.')

    def _average_errors(self, errors):
        """Returns average error."""
        raise NotImplementedError('"_average_errors" method should '
                                  'be implemented in a child class.')

    def _average_losses(self, losses):
        """Returns average loss."""
        raise NotImplementedError('"_average_losses" method should '
                                  'be implemented in a child class.')

    def _average_processing_time(self, processing_times):
        """Returns average processing time."""
        raise NotImplementedError('"_average_processing_time" method should '
                                  'be implemented in a child class.')

    def _report_training_progress(self):
        """Report current training progress after current epoch.

        The report, for example, may include training plot and log update.
        """
        raise NotImplementedError('"_report_training_progress" method should '
                                  'be implemented in a child class.')

    def _train_for_epoch(self,epoch_idx):
        """Returns training set losses."""
        self._network.train()
        self._current_losses = []
        number_of_batches = len(self._training_set_loader)
        Duration = 0
        cgvl = []
        for batch_index, batch in enumerate(self._training_set_loader):
            # break
            # if batch_index == 2:
            #     break
            # print("batch.size",batch['left']['event_queue'].size())
            if batch_index >= self._spec_title:
                 break
            # print("batch['left,image']",batch['left']['event_queue'].size())
            if _is_logging_required(batch_index, number_of_batches):
                
                self._logger.log('epoch {0:02d} ({1:02d}) : '
                                 'training: {2:05d} ({3:05d})'.format(
                                     self._current_epoch + 1, self._end_epoch,
                                     batch_index + 1, number_of_batches))
            
            if type(self._optimizer) == tuple:
                self._optimizer[0].zero_grad()
                self._optimizer[1].zero_grad()
            else:
                self._optimizer.zero_grad()
            
            # print("is_on_cuda",_is_on_cuda(self._network))
            if _is_on_cuda(self._network):
                batch = _move_tensors_to_cuda(batch)
            start_time = time.time()
            self._run_network(batch)

            # print('cm',self._network._temporal_aggregation._LTC_Conv.cm)
            # print('gleak',self._network._temporal_aggregation._LTC_Conv.gleak)
            # print('bias',self._network._regularization._upsample_to_fullsize.bias)
            Duration += time.time()-start_time
            self._compute_gradients_wrt_loss(batch)
            if type(self._optimizer) == tuple:
                self._optimizer[0].step()
                self._optimizer[1].step()
            else:
                self._optimizer.step()
            self._network._temporal_aggregation._LTC_Conv.apply_weight_constraints()
            self._current_losses.append(batch['loss'])
            # cgvl.append([self._network._temporal_aggregation._LTC_Conv.cm.detach().cpu().numpy(), self._network._temporal_aggregation._LTC_Conv.tau_m.detach().cpu().numpy(), self._network._temporal_aggregation._LTC_Conv.vleak.detach().cpu().numpy(),self._network._temporal_aggregation._LTC_Conv.E_revin.detach().cpu().numpy()])
            # cgvl.append([self._network._temporal_aggregation._LTC_Conv.cm.detach().cpu().numpy(), self._network._temporal_aggregation._LTC_Conv.gleak.detach().cpu().numpy(), self._network._temporal_aggregation._LTC_Conv.vleak.detach().cpu().numpy(),self._network._temporal_aggregation._LTC_Conv.E_revin.detach().cpu().numpy()])

            print('epoch: [{}/{}] batch: [{}/{}] time:{}'.format(
                                     self._current_epoch + 1, self._end_epoch,
                                     batch_index + 1, number_of_batches, time.time()-start_time))
            # if batch_index == 0:
            #     before_spade = batch['before_spade'].clone().detach().cpu().numpy()
            #     after_spade = batch['after_spade'].clone().detach().cpu().numpy()
            del batch
            th.cuda.empty_cache()
        # print("fps:{}".format(number_of_batches*4/Duration))

        # np.save(os.path.join(self._experiment_folder, 'ctv_ep%i'%(epoch_idx)), np.array(cgvl).squeeze())
        # if epoch_idx % 5 == 0:
        #     np.save(os.path.join(self._experiment_folder, 'left_spade_ep%i'%(epoch_idx)), before_spade)
        #     np.save(os.path.join(self._experiment_folder, 'right_ep%i'%(epoch_idx)), after_spade)

        return self._average_losses(self._current_losses)

    def _test(self):
        """Returns test set errors."""
        self._network.eval()
        self._current_errors = []
        self._current_processing_times = []
        number_of_examples = len(self._test_set_loader)
        debug_diserror = []
        debug_diserror_fidx = []
        example_indices = [100, 340, 980]
        # example_indices = [self._spec_title]
        for example_index, example in enumerate(self._test_set_loader):
            
            # if example['frame_index'] not in example_indices:
            #     continue
            print("example_index:[{:05d}/{:05d}]".format(example_index+1, number_of_examples))
            if _is_logging_required(example_index, number_of_examples):
                self._logger.log('epoch: {0:02d} ({1:02d}) : '
                         'validation: {2:05d} ({3:05d})'.format(
                             self._current_epoch + 1, self._end_epoch,
                             example_index + 1, number_of_examples))
            if _is_on_cuda(self._network):
                example = _move_tensors_to_cuda(example)
            with th.no_grad():
                
                if self._spec_title in [100,340, 980]:
                    # print('debug a')
                    # print(self._spec_title)
                    if example['frame_index'] == self._spec_title:

                        self._run_network_and_measure_time(example)
                        self._compute_error(example)
                        self._current_errors.append(example['error'])
                        self._current_processing_times.append(example['processing_time'])
                        self._visualize_example(example, example_index)
                        del example
                        th.cuda.empty_cache()
                    else:
                        pass    
                else:
                    
                    # print('debug b')
                    #if example['frame_index'] in [100,1700]: 
                    self._run_network_and_measure_time(example)
                    self._compute_error(example)
                    self._current_errors.append(example['error'])
                    # print(example['error']['mean_disparity_error'])
                    debug_diserror.append([example['network_output'].detach().cpu().numpy(), example['left']['disparity_image'].detach().cpu().numpy()])
                    debug_diserror_fidx.append(example['frame_index'].detach().cpu().numpy())
                    self._current_processing_times.append(example['processing_time'])
                    # self._visualize_example(example, example_index)
                    if example['frame_index'] in example_indices:
                        self._visualize_example(example, example_index)
                    del example
                    th.cuda.empty_cache()

        # np.save(os.path.join(self._experiment_folder, 'debug_diserror_val'), debug_diserror)
        # np.save(os.path.join(self._experiment_folder, 'debug_diserror_fidx_val'), debug_diserror_fidx)

        return (self._average_errors(self._current_errors),
                self._average_processing_time(self._current_processing_times))

    def test(self):
        """Test network and reports average errors and execution time."""
        self._initialize_filenames()
        self._logger = visualization.Logger(self._log_filename)
        average_errors, average_processing_time = self._test()
        self._report_test_results(average_errors, average_processing_time)
        return average_errors, average_processing_time


