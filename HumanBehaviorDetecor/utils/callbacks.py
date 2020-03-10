import json
import os

import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.platform import tf_logging as logging
from configuration import training_results_save_dir, test_picture_dir
from test_on_single_image import single_image_inference


class Mycheckpoint(ModelCheckpoint):
    def __init__(self, test_images_during_training=False, test_images_dir=None, test_images_freq='epoch', *args,
                 **kwargs):

        if test_images_during_training and test_images_dir is None:
            raise Exception('when testing images during training,test_images must not be None!')
        super(Mycheckpoint, self).__init__(*args, **kwargs)
        self.test_images_during_training = test_images_during_training
        self.test_images = test_images_dir
        self.test_images_freq = test_images_freq
        self._samples_seen_since_last_img_test = 0
        self.epochs_since_last_img_test = 0

    # def on_batch_end(self, batch, logs=None):
    #     super(Mycheckpoint, self).on_batch_end(batch, logs)
    #     if self.test_images_during_training:
    #         if isinstance(self.test_images_freq, int):
    #             self._samples_seen_since_last_img_test += logs.get('size', 1)
    #             if self._samples_seen_since_last_img_test >= self.test_images_freq:
    #                 self._visualize_training_results(batch=batch)
    #                 self._samples_seen_since_last_img_test = 0

    # def on_epoch_end(self, epoch, logs=None):
    #     super(Mycheckpoint, self).on_epoch_end(epoch, logs)
    #     self.epochs_since_last_img_test += 1
    #     if self.test_images_during_training:
    #         self._visualize_training_results(epoch=epoch)

    def _visualize_training_results(self, epoch=None, batch=None):
        # pictures : List of image directories.

        if isinstance(self.test_images_freq,
                      int) or self.epochs_since_last_img_test >= self.period:
            self.epochs_since_last_img_test = 0
            index = 0
            for picture in self.test_images:
                index += 1
                result = single_image_inference(image_dir=picture, model=self.model)
                if epoch is not None:
                    filename = training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index)
                elif batch is not None:
                    filename = training_results_save_dir + "batch-{}-picture-{}.jpg".format(batch, index)
                else:
                    filename = training_results_save_dir + "picture-{}.jpg".format(batch, index)
                cv2.imwrite(filename=filename, img=result)

    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}
        flag = True

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath_base = self._get_file_path(epoch, logs)

            if self.save_best_only:
                # 存储loss最小的模型用于预测

                filepath = filepath_base + 'best_epoch_{}'.format(epoch + 1)
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):

                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            try:
                                self.model.save_weights(filepath, overwrite=True)
                                with open(os.path.dirname(filepath) + '/training_cache.json', 'w',
                                          encoding='utf-8') as file_obj:
                                    # 存下模型存储时对应的epoch

                                    json.dump({'epoch': epoch + 1}, file_obj)
                                    print('saved epoch successfully!')

                            except Exception as e:
                                # flag = True
                                logging.warning('when trying to test data , exceptions \'{}\' happening!'.format(e))
                            try:
                                self._visualize_training_results(epoch=epoch)
                            except Exception as e:
                                logging.warning('when trying to test data , \'{}\' happening!'.format(e))
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))

            else:
                # 按存储频率来存模型，用于中断训练后继续训练模型
                filepath = filepath_base + "epoch-{}".format(epoch + 1)
                try:
                    with open(os.path.dirname(filepath) + '/training_cache.json', 'w', encoding='utf-8') as file_obj:
                        # 存下模型存储时对应的epoch
                        json.dump({'epoch': epoch + 1}, file_obj)
                except Exception as e:
                    logging.warning('when trying to write epoch to file exceptions {} happening!', e)
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                # 存模型的时候同时测试
                self._visualize_training_results(epoch=epoch)
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

            self._maybe_remove_file()
