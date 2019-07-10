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
                 epochs=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs

def main(train_obs, train_act, valid_obs, valid_act, args):
    trainer = Trainer(args)
    print("---------CREATING MODEL--------")
   
    #creating model
    model = build_model.build_model()

    # compiling model
    model.compile(
        optimizer = trainer.optimizer,
        loss = trainer.loss,
        metrics = trainer.metrics)

    print("----------TRAINING MODEL---------")
    # training model
    tr_history = model.fit(train_obs, train_act,
            epochs = trainer.epochs,
            verbose = 1,
            validation_data=(valid_obs,valid_act))

    return model

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    main(train_obs, train_act, valid_obs, valid_act, args)
