# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
# import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
from tqdm import tqdm
import random
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

seed = 0
torch.manual_seed(seed)  # Set seed for CPU
torch.cuda.manual_seed(seed)  # Set seed for current GPU
torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if using multi-GPU)
np.random.seed(seed)  # Set seed for numpy
random.seed(seed)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', '1'}:
        return True
    elif value.lower() in {'false', 'no', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--pt_file_path', type=str, default='DrugDesign/datasets/trainData/qm9.pt',
                    help='data pt file path')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', type=str2bool, default=True,
                      help='remove hydrogen')
parser.add_argument('--include_charges', type=str2bool, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)
atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

args.cuda = torch.cuda.is_available()
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch

    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

# Optionally update the training dataloader to use a DistributedSampler.
if dist.is_available() and dist.is_initialized():
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(dataloaders['train'].dataset, shuffle=True)
    dataloaders['train'] = torch.utils.data.DataLoader(
         dataloaders['train'].dataset,
         batch_size=args.batch_size,
         sampler=train_sampler,
         num_workers=args.num_workers
    )

data_dummy = next(iter(dataloaders['train']))

if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf

# Create EGNN flow.
# Note: We pass "None" for the device here and later move the model appropriately.
model, nodes_dist, prop_dist = get_model(args, None, dataset_info, dataloaders['train'])
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
optim = get_optim(args, model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl")  # Use NCCL backend for GPUs
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])  # Automatically assigned by torchrun
    torch.cuda.set_device(local_rank)
    return rank, local_rank

# def main():
if __name__ == "__main__":
    rank, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    print(f"Process {rank} using GPU {local_rank}")

    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'), map_location=device)
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'), map_location=device)
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Move model to device and wrap with DDP.
    model.to(device)
    model = DDP(model, device_ids=[local_rank])
    model_dp = model  # For compatibility in train/test functions.

    # Initialize EMA model if applicable.
    if args.ema_decay > 0:
        # Deepcopy the underlying module (unwrapping DDP) for EMA.
        model_ema = copy.deepcopy(model.module)
        ema = flow_utils.EMA(args.ema_decay)
        model_ema.to(device)
        model_ema = DDP(model_ema, device_ids=[local_rank])
    else:
        ema = None
        model_ema = model

    print(f'exp_name: {args.exp_name}')
    print(f'dataset: {args.dataset}')

    best_nll_val = 1e8
    best_nll_test = 1e8

    # Early stopping and validation settings.
    patience = 20
    adaptive_factor = 0.001
    epochs_no_improve = 0
    use_early_stop = False
    use_val = False

    for epoch in tqdm(range(args.start_epoch, args.n_epochs), desc=f"Rank {rank} Epochs"):
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
                    
        if rank == 0 and args.save_model:
            if epoch == args.n_epochs - 1:
                args.current_epoch = epoch + 1
                utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                utils.save_model(model.module if hasattr(model, "module") else model, 'outputs/%s/generative_model.npy' % args.exp_name)
                if args.ema_decay > 0:
                    utils.save_model(model_ema.module if hasattr(model_ema, "module") else model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                    pickle.dump(args, f)

            if epoch % 100 == 0:
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(model.module if hasattr(model, "module") else model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                if args.ema_decay > 0:
                    utils.save_model(model_ema.module if hasattr(model_ema, "module") else model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)    

        # if epoch > 1000 and epoch % 10 == 0 and use_val:
        #     nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema,
        #                    partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
        #                    property_norms=property_norms)
        #     min_delta = adaptive_factor * best_nll_val
        #     if nll_val < best_nll_val - min_delta:
        #         best_nll_val = nll_val
        #         epochs_no_improve = 0
        #         if args.save_model:
        #             args.current_epoch = epoch + 1
        #             utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
        #             utils.save_model(model.module if hasattr(model, "module") else model, 'outputs/%s/generative_model.npy' % args.exp_name)
        #             if args.ema_decay > 0:
        #                 utils.save_model(model_ema.module if hasattr(model_ema, "module") else model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
        #             with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
        #                 pickle.dump(args, f)
        #         if args.save_model:
        #             utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
        #             utils.save_model(model.module if hasattr(model, "module") else model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
        #             if args.ema_decay > 0:
        #                 utils.save_model(model_ema.module if hasattr(model_ema, "module") else model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
        #             with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
        #                 pickle.dump(args, f)
        #     else:
        #         epochs_no_improve += 1
        #     print(f'Val loss: {nll_val}')
        #     print(f'Best val loss: {best_nll_val}')
        #     if epochs_no_improve >= patience and use_early_stop:
        #         print(f"Early stopping triggered at epoch {epoch}")
        #         break

    print(f'Done - {args.exp_name}!')
    dist.destroy_process_group()

# if __name__ == "__main__":
#     main()
