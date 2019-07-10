from utils import parse_args
import importlib
import load_data
import gin
import build_model
import tensorflow as tf
from keras import losses

@gin.configurable
class Trainer(object):
    @gin.configurable
    def __init__(self, args,
                optimizer = None,
                 loss=None,
                 metrics=None,
                 batch_size=None,
                 epochs=None,
                 steps_per_epoch=None,
                 validation_steps=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

def print_info(trainer):
    print("----------TRAINING MODEL---------")
    print("optimizer = ", trainer.optimizer)
    print("optimizer learning rate = ", trainer.optimizer.lr)
    print("optimizer beta_1 = ", trainer.optimizer.beta_1)
    print("optimizer beta_2 = ", trainer.optimizer.beta_2)
    print("optimizer epsilon = ", trainer.optimizer.epsilon)
    print("optimizer decay = ", trainer.optimizer.decay)
    print("optimizer amsgrad = ", trainer.optimizer.amsgrad)
    print("loss = ", trainer.loss)
    print("metrics = ", trainer.metrics)
    print("batch_size = ", trainer.batch_size)
    print('epochs = ', trainer.epochs)
    print('steps_per_epoch = ', trainer.steps_per_epoch)
    print('validation_steps = ', trainer.validation_steps)


def main(train_obs, train_act, valid_obs, valid_act, args):
    trainer = Trainer(args) 
    #creating model
    model = build_model.build_model()
    
    # printing training info
    print_info(trainer)

    # compiling model
    model.compile(
        optimizer = trainer.optimizer,
        loss = trainer.loss,
        metrics = trainer.metrics)

    # training model
    tr_history = model.fit(train_obs, train_act,
            epochs = trainer.epochs,
            verbose = 1,
            validation_data=(valid_obs,valid_act),
            steps_per_epoch = trainer.steps_per_epoch,
            validation_steps = trainer.validation_steps)
            

    return model

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    main(train_obs, train_act, valid_obs, valid_act, args)
