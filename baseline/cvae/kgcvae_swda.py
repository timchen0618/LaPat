#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import KgCVAEConfig as Config
from data_apis.corpus import SWDADialogCorpus
from data_apis.data_utils import SWDADataLoader
from models.cvae import KgRnnCVAE

import torch

import glob

import pdb

# constants
tf.app.flags.DEFINE_string("word2vec_path", None, "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data/weibo_transformed_cvae_splitted.p", "Raw data directory.")
# tf.app.flags.DEFINE_string("data_dir", "data/test.p", "Raw data directory.")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_string("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", True, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("test_path", "try_5", "the dir to load checkpoint for forward only")
FLAGS = tf.app.flags.FLAGS

torch.cuda.set_device(1)


def get_checkpoint_state(ckp_dir):
    files = os.path.join(ckp_dir, "*.pth")
    files = glob.glob(files)
    files.sort(key=os.path.getmtime)
    return len(files) > 0 and files[-1] or None

def main():
    # config for training
    config = Config()

    # config for validation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 1

    # configuration for testing
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1
    test_config.backward_size = 2
    test_config.step_size = 2

    pp(config)

    # get data set
    api = SWDADialogCorpus(FLAGS.data_dir, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size)
    dial_corpus = api.get_dialog_corpus()
    meta_corpus = api.get_meta_corpus()

    train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")

    # convert to numeric input outputs that fits into TF models
    train_feed = SWDADataLoader("Train", train_dial, train_meta, config)
    valid_feed = SWDADataLoader("Valid", valid_dial, valid_meta, config)
    test_feed = SWDADataLoader("Test", test_dial, test_meta, config)

    if FLAGS.forward_only or FLAGS.resume:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run"+str(int(time.time())))

    # begin training
    if True:
        scope = "model"
        model = KgRnnCVAE(config, api, log_dir=None if FLAGS.forward_only else log_dir, scope=scope)

        print("Created computation graphs")
        # write config to a file for logging
        if not FLAGS.forward_only:
            with open(os.path.join(log_dir, "run.log"), "wb") as f:
                f.write(pp(config, output=False))

        # create a folder by force
        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = get_checkpoint_state(ckp_dir)
        print("Created models with fresh parameters.")
        model.apply(lambda m: [torch.nn.init.uniform_(p.data, -1.0 * config.init_w, config.init_w) for p in m.parameters()])

        # Load word2vec weight
        if api.word2vec is not None and not FLAGS.forward_only:
            print("Loaded word2vec")
            model.embedding.weight.data.copy_(torch.from_numpy(np.array(api.word2vec)))
        model.embedding.weight.data[0].fill_(0)

        if ckpt:
            print("Reading dm models parameters from %s" % ckpt)
            model.load_state_dict(torch.load(ckpt))

        # turn to cuda
        model.cuda()

        if not FLAGS.forward_only:
            log_dir = os.path.join(ckp_dir, 'log.txt')
            with open(log_dir, 'w') as write_log:

                dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ "-%d.pth")
                global_t = 1
                patience = 10  # wait for at least 10 epoch before stop
                dev_loss_threshold = np.inf
                best_dev_loss = np.inf
                for epoch in range(config.max_epoch):
                    print(">> Epoch %d with lr %f" % (epoch, model.learning_rate))

                    # begin training
                    
                    if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                        train_feed.epoch_init(config.batch_size, config.backward_size,
                                            config.step_size, shuffle=True)
                    global_t, train_loss = model.train_model(global_t, train_feed, update_limit=config.update_limit)
                    
                    # begin validation
                    valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                        valid_config.step_size, shuffle=False, intra_shuffle=False)
                    model.eval()
                    valid_loss = model.valid_model("ELBO_VALID", valid_feed)
                    '''
                    test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                        test_config.step_size, shuffle=True, intra_shuffle=False)
                    model.test_model(test_feed, num_batch=None)
                    '''
                    model.train()

                    loss_line = "Epoch %d, train loss = %f, valid loss = %f" % (epoch, train_loss, valid_loss)
                    print(loss_line)
                    write_log.write(loss_line)
                    write_log.write('\n')

                    done_epoch = epoch + 1
                    # only save a models if the dev loss is smaller
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if config.op == "sgd" and done_epoch > config.lr_hold:
                        model.learning_rate_decay()

                    if valid_loss < best_dev_loss:
                        if valid_loss <= dev_loss_threshold * config.improve_threshold:
                            patience = max(patience, done_epoch * config.patient_increase)
                            dev_loss_threshold = valid_loss

                        # still save the best train model
                        if FLAGS.save_model:
                            print("Save model!!")
                            torch.save(model.state_dict(), dm_checkpoint_path %(epoch))
                        best_dev_loss = valid_loss


                    if config.early_stop and patience <= done_epoch:
                        print("!!Early stop due to run out of patience!!")
                        break
                print("Best validation loss %f" % best_dev_loss)
                print("Done training")
        else:
            # begin validation
            # begin validation
            model.eval()
            '''
            valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            model.eval()
            model.valid_model("ELBO_VALID", valid_feed)

            test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            model.valid_model("ELBO_TEST", test_feed)
            '''

            dest_f = open(os.path.join(log_dir, "test.txt"), "wb")
            test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                 test_config.step_size, shuffle=False, intra_shuffle=False)
            model.test_model(test_feed, num_batch=None, repeat=5, dest=dest_f)
            # model.test_model(test_feed, num_batch=None, repeat=1)
            # model.train()
            dest_f.close()

if __name__ == "__main__":
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()













