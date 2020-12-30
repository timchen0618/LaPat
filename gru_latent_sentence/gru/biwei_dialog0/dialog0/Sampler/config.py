# Save model path
out_dir = "./out/sampler"

# Hyperparameters
hidden_dim = 500
emb_dim = 620
num_train_epochs = 30
start_epoch_at = None
steps_per_stats = 100
batch_size = 128
beam_size = 4
vocab_size = 50000

optim_method = 'adam'
lr = 0.002
learning_rate_decay = 0.5
weight_decay = 0.000001
start_decay_at = None
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_cuda = True

lr_coverage = 0.15
