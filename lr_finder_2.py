import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LRFind(tf.keras.callbacks.Callback): 
    def __init__(self, min_lr, max_lr, n_rounds): 
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lrs = []
        self.losses = []
        self.lrs_history = []
        self.losses_history = []
        self.n_rounds = n_rounds
        self.best_lr = 0

    def on_train_begin(self, logs=None):
      self.model.optimizer.lr = self.min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch %2 == 0:
          self.weights = self.model.get_weights()
          self.lrs = []
          self.losses = []
          self.step_up = (self.max_lr / self.min_lr) ** (1 / self.n_rounds)

        else:
          self.step_up = 1
          self.model.optimizer.lr = self.best_lr
          print(f"Best LR: {self.best_lr}")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
          self.model.set_weights(self.weights)
          self.best_lr = self.lrs[np.argmin(self.losses)]*1e-1
          self.lrs_history.append(self.lrs)
          self.losses_history.append(self.losses)

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs["loss"])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
 
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True
  
		
