import argparse
import deepspeed


def add_args(parser):
    # model architecture options
    parser.add_argument('--input_dim', type=int, default=3,
                        help='Number of input dimensions (3 for 3D point clouds)')
    parser.add_argument('--max_outputs', type=int, default=2500,
                        help='Number of maximum output elements')
    parser.add_argument('--init_dim', type=int, default=64,
                        help='Number of dimensions for each initial set element')
    parser.add_argument('--n_mixtures', type=int, default=16,
                        help='Number of mixture components for the initial set')
    parser.add_argument('--fixed_gmm', action='store_true',
                        help='Whether to use fixed initialization (Fibonacci sphere-based) for initial set GMM parameters')
    parser.add_argument('--train_gmm', action='store_true',
                        help='Whether to train initial set GMM parameters via reparameterization')
    parser.add_argument('--z_dim', type=int, default=16,
                        help='Number of dimensions for each latent set element')
    parser.add_argument('--z_scales', nargs='+', type=int, default=[2, 4, 8, 16],
                        help='Top-down scales for hierarchical latent sets')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Number of hidden dimensions')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--slot_att', action='store_true',
                        help='Whether to use slot attention')
    parser.add_argument('--i_net', type=str, default='elem_mlp',
                        help='Induced network to use', choices=['full_mlp', 'elem_mlp', 'set_transformer'])
    parser.add_argument('--i_net_layers', type=int, default=0,
                        help='Number of hidden layers in induced network')
    parser.add_argument('--d_net', type=str, default='set_transformer',
                        help='Deterministic layer to use', choices=['elem_mlp', 'set_transformer'])
    parser.add_argument('--enc_in_layers', type=int, default=0,
                        help='Number of deterministic layers in pre-encoder')
    parser.add_argument('--dec_in_layers', type=int, default=0,
                        help='Number of deterministic layers in pre-decoder')
    parser.add_argument('--dec_out_layers', type=int, default=0,
                        help='Number of deterministic layers in post-decoder')
    parser.add_argument('--isab_inds', type=int, default=16,
                        help='Number of inducing points in deterministic layers')
    parser.add_argument('--ln', action='store_true',
                        help='Whether to use layer normalization')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function for MLP', choices=['relu', 'tanh'])
    parser.add_argument('--use_bn', action='store_true',
                        help='Whether to use batchnorm for MLP.')
    parser.add_argument('--residual', action='store_true',
                        help='Whether to use residual connections for MLP.')

    # training options
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use', choices=['adam', 'adamax', 'sgd'])
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--max_grad_norm', default=5., type=float,
                        help='Gradient norm clipping')
    parser.add_argument('--max_grad_threshold', default=None, type=float,
                        help='Gradient norm threshold for update')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--dropout_p', type=float, default=0.,
                        help='Dropout rate.')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Total epochs to train')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--matcher', default='chamfer', type=str, choices=('hungarian', 'chamfer', 'all'),
                        help='Matcher for reconstruction loss computation')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='KL loss weight')
    parser.add_argument('--kl_warmup_epochs', default=0, type=int,
                        help='KL annealing epochs')
    parser.add_argument('--scheduler', type=str, default='none',
                        help='Type of learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Length of learning rate warm-up')
    parser.add_argument('--exp_decay', type=float, default=1.,
                        help='Learning rate schedule exponential decay rate')
    parser.add_argument('--exp_decay_freq', type=int, default=1,
                        help='Learning rate exponential decay frequency')

    # data options
    parser.add_argument('--dataset_type', default='shapenet15k', type=str,
                        help='Dataset to train on, one of ShapeNet / MNIST / MultiMNIST',
                        choices=['shapenet15k', 'mnist', 'multimnist'])
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading threads')
    # ShapeNet options
    parser.add_argument('--shapenet_data_dir', default='/data/shapenet/ShapeNetCore.v2.PC15k', type=str,
                        help='Path to training ShapeNet data')
    parser.add_argument('--cates', nargs='+', default=['airplane'],
                        help='List of object categories to construct train/val dataset')
    parser.add_argument('--dataset_scale', type=float, default=1.,
                        help='Scale of the dataset (x,y,z * scale = real output, default=1).')
    parser.add_argument('--normalize_per_shape', action='store_true',
                        help='Whether to perform normalization per shape.')
    parser.add_argument('--normalize_std_per_axis', action='store_true',
                        help='Whether to perform normalization per axis.')
    parser.add_argument('--denormalized_loss', action='store_true',
                        help='Whether to perform denormalization before loss computation.')
    parser.add_argument("--tr_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (train)')
    parser.add_argument("--te_max_sample_points", type=int, default=2048,
                        help='Max number of sampled points (test)')
    parser.add_argument("--standardize_per_shape", action='store_true',
                        help='Whether to perform standardization per shape')
    parser.add_argument("--eval_with_train_offset", action='store_true',
                        help='Whether to add train offset, for debugging purpose only')
    # MNIST options
    parser.add_argument('--digits', nargs='+', default=None,
                        help='Digit to filter, None to use all')
    parser.add_argument('--threshold', default=0., type=float,
                        help='Pixel binarization threshold')
    parser.add_argument('--mnist_data_dir', default='cache/mnist', type=str,
                        help='Path to MNIST image data')
    parser.add_argument('--mnist_cache', default=None, type=str,
                        help='Directory to cache MNIST image data')
    parser.add_argument('--multimnist_data_dir', default='cache/multimnist', type=str,
                        help='Path to MultiMNIST image data')
    parser.add_argument('--multimnist_cache', default=None, type=str,
                        help='Directory to cache MultiMNIST image data')

    # logger options
    parser.add_argument('--log_name', default=None, type=str, help="Name for the log dir")
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--viz_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_recon_only', action='store_true',
                        help='Whether to perform reconstruction for validation.')

    # validation options
    parser.add_argument('--no_validation', action='store_true',
                        help='Whether to disable validation altogether.')
    parser.add_argument('--save_val_results', action='store_true',
                        help='Whether to save the validation results.')
    parser.add_argument('--no_eval_sampling', action='store_true',
                        help='Whether to evaluate sampling.')
    parser.add_argument('--max_validate_shapes', type=int, default=None,
                        help='Max number of shapes used for validation pass.')

    # resume options
    parser.add_argument('--resume', action='store_true',
                        help='resume training from loaded checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to the checkpoint to be loaded. Should end with .pth')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Whether to resume the optimizer when resumed training.')
    parser.add_argument('--resume_non_strict', action='store_true',
                        help='Whether to resume in none-strict mode.')
    parser.add_argument('--resume_dataset_mean', type=str, default=None,
                        help='Path to the file storing the dataset mean.')
    parser.add_argument('--resume_dataset_std', type=str, default=None,
                        help='Path to the file storing the dataset std.')

    # distributed training
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # evaluation options
    parser.add_argument('--eval', action='store_true',
                        help='Whether to evaluate the loaded checkpoint')
    parser.add_argument('--bn_mode', default='eval', type=str, choices=('train', 'eval'),
                        help='Test time BatchNorm mode')
    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='VAE Set Generation Experiment')
    parser = deepspeed.add_config_arguments(parser)
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
