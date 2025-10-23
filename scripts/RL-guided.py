


import torch, random, copy, os
import torch.nn as nn
import numpy as np
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, assert_correctly_masked
from torch.optim.lr_scheduler import CosineAnnealingLR
from equivariant_diffusion import utils as diffusion_utils

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import utils

from torch.cuda.amp import autocast, GradScaler

# Set random seeds
seeds = 2024
torch.manual_seed(seeds)
torch.cuda.manual_seed(seeds)
torch.cuda.manual_seed_all(seeds)
np.random.seed(seeds)
random.seed(seeds)


# import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import seaborn as sns

import torch
import torch.nn as nn
import numpy as np
from qm9 import dataset
# from qm9.models import get_model
from qm9.utils import compute_mean_mad, prepare_context
from qm9.analyze import analyze_stability_for_molecules
from configs.datasets_config import get_dataset_info
import pickle
from os.path import join


# -------------------------------------------- start -------------------------------------------------

import torch, math
from torch.distributions.categorical import Categorical

import numpy as np
from egnn.models import EGNN_dynamics_QM9

from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from scipy.stats import beta
import pandas as pd

def get_dist(data_path, quantile = 50):
    df = pd.read_csv(data_path)
    qed_cutoff = np.percentile(df['qed'], quantile)
    sas_cutoff = np.percentile(df['sas'], 100-quantile)
    aff_cutoff = np.percentile(df['affinity'], 100-quantile)

    qualified = df[
        (df['qed'] > qed_cutoff) &
        (df['sas'] < sas_cutoff) &
        (df['affinity'] < aff_cutoff)
    ]

    return qualified

def list_ave(my_list):
    my_list = [x if math.isfinite(x) else 0 for x in my_list]
    return sum(my_list) / len(my_list)

def get_model(args, device, dataset_info, dataloader_train, mode='baseline', RL=False):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning, mode=mode)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    if args.probabilistic_model == 'diffusion':

        if RL:
            vdm = EnVariationalDiffusionPPO(
                dynamics=net_dynamics,
                in_node_nf=in_node_nf,
                n_dims=3,
                timesteps=args.diffusion_steps,
                noise_schedule=args.diffusion_noise_schedule,
                noise_precision=args.diffusion_noise_precision,
                loss_type=args.diffusion_loss_type,
                norm_values=args.normalize_factors,
                include_charges=args.include_charges
                )
        else:
            vdm = EnVariationalDiffusion(
                dynamics=net_dynamics,
                in_node_nf=in_node_nf,
                n_dims=3,
                timesteps=args.diffusion_steps,
                noise_schedule=args.diffusion_noise_schedule,
                noise_precision=args.diffusion_noise_precision,
                loss_type=args.diffusion_loss_type,
                norm_values=args.normalize_factors,
                include_charges=args.include_charges
                )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None, mode=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties

        size = len(dataloader.dataset.data[list(dataloader.dataset.data.keys())[0]])
        data_path = f'/data/lab_ph/kyle/projects/DrugDesign/data/testset.csv'
        qualified = get_dist(data_path, quantile = 50)[:size] # can be changed
        self.conditions_tensor = {
            'qed': torch.tensor(qualified['qed'].tolist(), dtype=torch.float32),
            'sas': torch.tensor(qualified['sas'].tolist(), dtype=torch.float32),
            'affinity': torch.tensor(qualified['affinity'].tolist(), dtype=torch.float32)
        }

        for prop in properties:
            self.distributions[prop] = {}
            if mode == 'baseline':
                self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                    dataloader.dataset.data[prop],
                                    self.distributions[prop])
            else:
                self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                    self.conditions_tensor[prop],
                                    self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            # dist = self.distributions[prop][n_nodes]
            if n_nodes not in self.distributions[prop]:
                available_keys = list(self.distributions[prop].keys())
                closest_n = min(available_keys, key=lambda x: abs(x - n_nodes))
                print(f"Warning: node count {n_nodes} not found for {prop}. Using {closest_n} instead.")
                dist = self.distributions[prop][closest_n]
            else:
                dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


# --------------------------------------------- end -------------------------------------------------


from torch.nn import functional as F

def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def compute_pairwise_distances(x, y):
    """
    x: (n, d)
    y: (m, d)
    return: (n, m) pairwise squared distances
    """
    x_norm = (x ** 2).sum(dim=1).unsqueeze(1)  # (n, 1)
    y_norm = (y ** 2).sum(dim=1).unsqueeze(0)  # (1, m)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return dist

def mmd_rbf(X, Y, sigma=1.0):
    """
    Per-sample MMD reward version.
    X: (n, d) generated molecules
    Y: (m, d) reference molecules
    return: (n,) reward vector
    """
    beta = 1.0 / (2.0 * sigma ** 2)

    # Compute K_XY (each sample of X to all Y)
    dist_XY = compute_pairwise_distances(X, Y)
    K_XY = torch.exp(-beta * dist_XY)  # (n, m)

    # Compute K_YY once
    dist_YY = compute_pairwise_distances(Y, Y)
    K_YY = torch.exp(-beta * dist_YY)
    m = Y.size(0)
    K_YY_sum = (K_YY.sum() - m) / (m * (m - 1))  # exclude diagonal

    # MMD per sample
    mmd_per_sample = K_XY.mean(dim=1) - K_YY_sum
    return -mmd_per_sample

class GlobalStatTracker:
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.buffer = []

    def update(self, rewards):
        self.buffer.extend(rewards.tolist())
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        mean = torch.tensor(self.buffer).mean()
        std = torch.tensor(self.buffer).std() + 1e-8
        advantages = (rewards - mean) / std
        return advantages

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_score = -float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, total_reward):
        if total_reward > self.best_score + self.delta:
            self.best_score = total_reward
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def str2bool(value):
  if isinstance(value, bool):
    return value
  if value.lower() in {'true', 'yes', '1'}:
    return True
  elif value.lower() in {'false', 'no', '0'}:
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def compute_cutoffs(property_dict, quantile=50):

    high_good = {'qed': True, 'sas': False, 'affinity': False}
    
    cutoff_dict = {}
    for prop, values in property_dict.items():

        values = np.array(values)
        values = values[np.isfinite(values)]

        if len(values) == 0: # set based on needs
            if prop == 'qed': cutoff = 0.5   
            elif prop == 'sas': cutoff = 8.0
            elif prop == 'affinity': cutoff = -6.0
        else:
            if high_good.get(prop, True):
                cutoff = np.percentile(values, 100 - quantile)
            else:
                cutoff = np.percentile(values, quantile)
        cutoff_dict[prop] = cutoff
    
    return cutoff_dict

class EnVariationalDiffusionPPO(EnVariationalDiffusion):
    def __init__(self, dynamics, in_node_nf, n_dims, timesteps=1000, 
                 parametrization='eps', noise_schedule='learned', 
                 noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.), 
                 norm_biases=(None, 0., 0.), include_charges=True):
        super().__init__(dynamics, in_node_nf, n_dims, timesteps, parametrization,
                         noise_schedule, noise_precision, loss_type, norm_values,
                         norm_biases, include_charges)

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # print(f"sample_p_zs_given_zt: eps_t stats at t={t[0].item()}: {eps_t.mean().item(), eps_t.std().item()}")

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        # if t[0][0].cpu() == 0.9980: print('old:', t, mu)
        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # ------------------------------------
        sigma = torch.clamp(sigma, min=1e-4)
        with torch.no_grad():
            zs_detached = zs.detach()
        log_prob = (
                -((zs_detached - mu) ** 2) / (2 * torch.clamp(sigma**2, min=1e-12))
                - torch.log(sigma)
                - 0.5 * torch.log(2.0 * torch.pi * torch.ones(1, device=mu.device))
            )
        
        log_prob = log_prob.mean(dim=(-1, -2))

        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():

            log_prob = (log_prob * node_mask).sum(dim=(-1, -2)) / node_mask.sum(dim=(-1, -2))
            log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
            
        # ------------------------------------

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        
        return zs, log_prob, (mu, sigma) ##

    # @torch.no_grad()
    def sample_with_trajectory(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):

        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
        
        batch = {'s':[], 't':[], 'node_mask':None, 'edge_mask':None, 'context':None, 'mu_sigma':[]}  ##
        latents = [] 
        next_latents = []  
        log_probs = []
        
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        batch['node_mask'] = node_mask
        batch['edge_mask'] = edge_mask
        batch['context'] = context

        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            batch['s'].append(s_array)
            batch['t'].append(t_array)
            latents.append(z)

            z, log_prob, (mu, sigma) = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

            log_probs.append(log_prob)
            next_latents.append(z)
            batch['mu_sigma'].append((mu, sigma))  

        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                    f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return latents, next_latents, log_probs, batch, (x, h)  

    def do_sample(self, args, device, dataset_info, prop_dist=None, nodesxsample=torch.tensor([10]), 
                  context=None, fix_noise=False):
        
        max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

        assert int(torch.max(nodesxsample)) <= max_n_nodes
        batch_size = len(nodesxsample)

        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1

        # Compute edge_mask

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)

        # TODO FIX: This conditioning just zeros.
        if args.context_node_nf > 0:
            if context is None:
                context = prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        else:
            context = None

        if args.probabilistic_model == 'diffusion':
            latents, next_latents, log_probs, batch, (x, h) = self.sample_with_trajectory(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

            assert_correctly_masked(x, node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            one_hot = h['categorical']
            charges = h['integer']

            assert_correctly_masked(one_hot.float(), node_mask)
            if args.include_charges:
                assert_correctly_masked(charges.float(), node_mask)

        else:
            raise ValueError(args.probabilistic_model)

        return one_hot, charges, x, node_mask, (latents, next_latents, log_probs, batch)

    def get_trajectory(self, n_samples, batch_size, dataset_info, hyper_paras=None):

        batch_size = min(batch_size, n_samples)
        assert n_samples % batch_size == 0
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        latents_n, next_latents_n, log_probs_n = [], [], []  
        for i in tqdm(range(int(n_samples/batch_size))):
            nodesxsample = nodes_dist.sample(batch_size)
            one_hot, charges, x, node_mask, (latents, next_latents, log_probs, batch) = self.do_sample(
                    args, device, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample
                )
            latents_n.extend(latents)
            next_latents_n.extend(next_latents)
            log_probs_n.extend(log_probs)
            # batch_n.extend(batch)
            batch_n = batch

            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
        stability_dict, rdkit_metrics = analyze_stability_for_molecules(molecules, dataset_info, RL=True, k=hyper_paras['diversity_k'], hyper_paras=hyper_paras)

        if rdkit_metrics is not None:
            rdkit_metrics = rdkit_metrics[0]
            validity, uniqueness, novelty, mean_qed, mean_sas, mean_affinity, mean_diversity, [qed_list, sas_list, affinity_list, diversity_list, fitness_list, valid_list, unique_list, novel_list] = rdkit_metrics

            atm_stable = stability_dict['atm_stable']
            mol_stable = stability_dict['mol_stable']

            qed_tensor = torch.tensor(qed_list, dtype=torch.float32, device=device)
            qed_normalized = qed_tensor

            sas_tensor = torch.tensor(sas_list, dtype=torch.float32, device=device)
            mask = torch.isfinite(sas_tensor)
            sas_normalized = torch.full_like(sas_tensor, float('-inf'))
            sas_normalized[mask] = (10 - torch.clamp(sas_tensor[mask], max=10)) / 9

            affinity_tensor = torch.tensor(affinity_list, dtype=torch.float32, device=device)
            mask = torch.isfinite(affinity_tensor)
            affinity_normalized = torch.full_like(affinity_tensor, float('-inf'))
            # affinity_normalized[mask] = 1 - torch.clamp(affinity_tensor[mask] + 20, 0, 20) / 20
            affinity_max = hyper_paras['affinity_max']
            affinity_normalized[mask] = 1 - torch.clamp(affinity_tensor[mask] + affinity_max, 0, affinity_max) / affinity_max

            diversity_tensor = torch.tensor(diversity_list, dtype=torch.float32, device=device)
            diversity_normalized = diversity_tensor

            if set(fitness_list) == {0}:
                print("manually assigned fitness")
                prop_dict_ = {'qed' : qed_list, 'sas' : sas_list, 'affinity': affinity_list,}
                cutoff = compute_cutoffs(prop_dict_, quantile=50)
                qed_mask = (np.array(qed_list) >= cutoff['qed']).astype(int)
                sas_mask = (np.array(sas_list) <= cutoff['sas']).astype(int)
                affinity_mask = (np.array(affinity_list) <= cutoff['affinity']).astype(int)
                fitness_list = (qed_mask * sas_mask * affinity_mask).tolist()

            fitness_tensor = torch.tensor(fitness_list, dtype=torch.float32, device=device)
            fitness_normalized = fitness_tensor
            mean_fitness = np.mean([x for x in fitness_list if x != float('0.0')])

            print(f'Validity: {validity*100:.2f}%, Uniqueness: {uniqueness*100:.2f}%, Novelty: {novelty*100:.2f}%, Atom_stability: {atm_stable*100:.2f}%, Molecule_stability: {mol_stable*100:.2f}%')
            print(f"exclude zeros: mean qed: {mean_qed:.2f}, mean sas: {mean_sas:.2f}, mean affinity: {mean_affinity:.2f}, mean diversity: {mean_diversity:.2f}, mean fitness: {mean_fitness:.2f}")
            print(f"include zeros: mean qed: {list_ave(qed_list):.2f}, mean sas: {list_ave(sas_list):.2f}, mean affinity: {list_ave(affinity_list):.2f}, mean diversity: {list_ave(diversity_list):.2f}, mean fitness: {list_ave(fitness_list):.2f}")
            
            merged_list = []
            target_normalized = {
                'qed':qed_normalized,
                'sas':sas_normalized,
                'affinity':affinity_normalized,
                'diversity':diversity_normalized,
                'fitness':fitness_normalized,
            }

            for task in hyper_paras['tasks']:  
                merged_list.append(target_normalized[task])

            reward_vector = torch.stack(merged_list, dim=1)

            bonus = torch.ones(batch_size, device=device)
            for i in range(batch_size):

                if valid_list[i]:
                    bonus[i] = hyper_paras['valid_bonus'] 
                if unique_list[i]:
                    bonus[i] = hyper_paras['unique_bonus'] 
                if novel_list[i]:
                    bonus[i] = hyper_paras['novel_bonus'] 
            
            reward_vector = reward_vector * bonus.unsqueeze(1)
            reward_vector[~torch.isfinite(reward_vector)] = 0
        else:
            reward_vector = torch.zeros(batch_size, len(merged_list), device=device)
        
        return latents_n, next_latents_n, log_probs_n, batch_n, reward_vector, [validity*100, uniqueness*100, novelty*100]


    def get_new_log_prob(self, s, t, zt, node_mask, edge_mask, context, old_next_latent=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # print(f"get_new_log_prob: eps_t stats at t={t[0].item()}: {eps_t.mean().item(), eps_t.std().item()}")

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        sigma = sigma_t_given_s * sigma_s / sigma_t

        sigma = torch.clamp(sigma, min=1e-4)
        with torch.no_grad():
            # zs_detached = zs.detach()
            old_next_latent_detach = old_next_latent.detach()
        log_prob = (
                -((old_next_latent_detach - mu) ** 2) / (2 * torch.clamp(sigma**2, min=1e-12))
                - torch.log(sigma)
                - 0.5 * torch.log(2.0 * torch.pi * torch.ones(1, device=mu.device))
            )
        
        log_prob = log_prob.mean(dim=(-1, -2))
        # print(log_prob)

        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():

            log_prob = (log_prob * node_mask).sum(dim=(-1, -2)) / node_mask.sum(dim=(-1, -2))
            log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)

        # raise False
        # ------------------------------------

        return log_prob, mu, sigma

def compute_angle_matrix(grads: torch.Tensor):
    """
    grads: Tensor of shape (T, D)
    returns: angle matrix of shape (T, T) in degrees
    """
    T = grads.shape[0]
    angle_matrix = torch.zeros((T, T))

    for i in range(T):
        for j in range(T):
            cos_sim = F.cosine_similarity(grads[i], grads[j], dim=0)
            angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180 / torch.pi
            angle_matrix[i, j] = angle

    return angle_matrix

def plot_task_gradients_2d(grads: torch.Tensor, task_labels=None):
    """
    grads: (T, D) tensor
    """
    grads_np = grads.cpu().numpy()
    pca = PCA(n_components=2)
    grads_2d = pca.fit_transform(grads_np)

    plt.figure(figsize=(6, 6))
    for i in range(len(grads_2d)):
        x, y = grads_2d[i]
        plt.arrow(0, 0, x, y, head_width=0.05, length_includes_head=True, color='C{}'.format(i))
        if task_labels:
            plt.text(x * 1.1, y * 1.1, task_labels[i], fontsize=12)
        else:
            plt.text(x * 1.1, y * 1.1, f'Task {i}', fontsize=12)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Task Gradient Directions (PCA 2D)")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.show()
    out_png = '/data/lab_ph/kyle/projects/DrugDesign/baselines/rl_gym/angles.png'
    plt.savefig(out_png, bbox_inches="tight", dpi=200)

def plot_task_gradients_3d(grads: torch.Tensor, task_labels=None):
    """
    grads: Tensor of shape (T, D)
    """
    grads_np = grads.cpu().numpy()
    pca = PCA(n_components=3)
    grads_3d = pca.fit_transform(grads_np)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    origin = [0, 0, 0]
    for i in range(len(grads_3d)):
        x, y, z = grads_3d[i]
        ax.quiver(*origin, x, y, z, length=1.0, normalize=True, color=f'C{i}')
        if task_labels:
            ax.text(x * 1.4, y * 1.4, z * 1.4, task_labels[i], fontsize=12)
        else:
            ax.text(x * 1.4, y * 1.4, z * 1.4, f'Task {i}', fontsize=12)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Task Gradient Directions (PCA 3D)")
    # ax.quiver(*origin, x, y, z, length=1.0, normalize=True, color=f'C{i}')
    # ax.text(x * 1.4, y * 1.4, z * 1.4, task_labels[i], fontsize=12)
    plt.tight_layout()
    # plt.show()
    out_png = '/data/lab_ph/kyle/projects/DrugDesign/baselines/rl_gym/angles.png'
    plt.savefig(out_png, bbox_inches="tight", dpi=200)

def compute_angle_matrix(grads: torch.Tensor):
    T = grads.shape[0]
    angle_matrix = torch.zeros((T, T))

    for i in range(T):
        for j in range(T):
            cos_sim = F.cosine_similarity(grads[i], grads[j], dim=0)
            angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180 / torch.pi
            angle_matrix[i, j] = angle

    return angle_matrix.numpy()

def plot_combined(grads: torch.Tensor, task_labels=None):
    grads_np = grads.cpu().numpy()
    pca = PCA(n_components=3)
    grads_3d = pca.fit_transform(grads_np)
    angle_matrix = compute_angle_matrix(grads)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    sns.heatmap(angle_matrix, annot=True, fmt=".1f", cmap="coolwarm",
                xticklabels=task_labels, yticklabels=task_labels, ax=ax1)
    ax1.set_title("Task Gradient Angle Matrix (°)")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    origin = [0, 0, 0]
    for i in range(len(grads_3d)):
        x, y, z = grads_3d[i]
        ax2.quiver(*origin, x, y, z, length=1.0, normalize=True, color=f'C{i}')
        label = task_labels[i] if task_labels else f'Task {i}'
        ax2.text(x * 1.4, y * 1.4, z * 1.4, label, fontsize=12)

    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("Task Gradient Directions (PCA 3D)")

    plt.tight_layout()
    out_png = '/data/lab_ph/kyle/projects/DrugDesign/baselines/rl_gym/combined.png'
    plt.savefig(out_png, bbox_inches="tight", dpi=200)

def ppo_update(generative_model, optimizer, scheduler, latents_n, next_latents_n, log_probs_n, batch_n, advantages, reward_vector=None, old_reward_vector=None, scalar_rewards=None, weights=None, device='cuda', clip_param=1e-4, scaler=None, episode=0, num_episodes=1000, hyper_paras=None, args=None, eta=0.05):

    k = hyper_paras['t_indices'] 
    num_total_steps = len(log_probs_n)
    t_indices = torch.randperm(num_total_steps)[:k].tolist()

    optimizer.zero_grad()

    for t in t_indices:
        
        generative_model.eval()
        new_log_prob, new_mu, new_sigma = generative_model.get_new_log_prob(
            batch_n['s'][t], batch_n['t'][t], latents_n[t], 
            batch_n['node_mask'], batch_n['edge_mask'], batch_n['context'], 
            old_next_latent=next_latents_n[t]
        )
        generative_model.train()

        old_log_prob = log_probs_n[t]
        ratio = torch.exp(new_log_prob - old_log_prob)

        old_mu, old_sigma = batch_n['mu_sigma'][t]
        kl_coef = hyper_paras['kl_coef'] 
        kl_penalty = 0.5 * (
            torch.log(new_sigma / old_sigma)
            + (old_sigma**2 + (old_mu - new_mu)**2) / (2 * new_sigma**2)
            - 0.5
        ).mean()

        entropy_coef = hyper_paras['entropy_coef'] * max(0, 1 - episode / num_episodes) 
        entropy = torch.log(new_sigma + 1e-6).mean()
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
        step_loss = -torch.min(surr1, surr2).mean() + kl_coef * kl_penalty - entropy_coef * entropy

        step_loss.backward(retain_graph=(t != t_indices[-1]))


    torch.nn.utils.clip_grad_norm_(generative_model.parameters(), max_norm=0.05)
    optimizer.step()

    if hyper_paras['use_scheduler'] == True:
        scheduler.step()
    

def train_ppo_(generative_model, n_samples, batch_size, dataset_info=None, num_episodes=1000, device='cuda', stat_tracker=None, hyper_paras=None, args=None):
    
    optimizer = torch.optim.AdamW(generative_model.parameters(), lr=hyper_paras['rl_lr'], weight_decay=1e-2)  # 1e-5
    scheduler = CosineAnnealingLR(optimizer, T_max=num_episodes, eta_min=1e-6)
    scaler = GradScaler()
    init_reward_vector = None

    if hyper_paras['manual_weights'] != None:
        weights = torch.tensor(hyper_paras['manual_weights']).to(device) 
    else:
        weights = None

    original_reward = float('-inf')
    original_validity = float('-inf')

    early_stopper = EarlyStopping(patience=50)

    for episode in range(num_episodes):

        old_generative_model = copy.deepcopy(generative_model).to(device)
        old_generative_model.eval()

        with torch.no_grad():
            latents_n, next_latents_n, log_probs_n, batch_n, reward_vector, [validity, uniqueness, novelty] = old_generative_model.get_trajectory(n_samples, batch_size, dataset_info, hyper_paras=hyper_paras)

        eta = 0.05
        if weights == None:
            num_tasks = len(hyper_paras['tasks'])
            weights = torch.ones(num_tasks, device=device) / num_tasks

        if hyper_paras['dynamic_weight'] and init_reward_vector != None:   
            delta = reward_vector.mean(dim=0) - init_reward_vector.mean(dim=0)
            delta = torch.clamp(delta, min=1e-6)
            importance = 1 / (delta + 1e-6) 
            new_w = importance / (importance.sum() + 1e-8)
            weights = (1 - eta) * weights + eta * new_w

        if stat_tracker != None:
            scalar_rewards = (reward_vector * weights).sum(dim=1)  
            advantages = stat_tracker.update(scalar_rewards) if stat_tracker else (scalar_rewards - scalar_rewards.mean()) / (scalar_rewards.std() + 1e-8)
        else:
            advantages = (reward_vector - reward_vector.mean(dim=0)) / (reward_vector.std(dim=0) + 1e-8)  ## smooth ?

        advantages = torch.clamp(advantages, min=-hyper_paras['adv_clip'], max=hyper_paras['adv_clip']).to(device)

        avg_rewards = reward_vector.mean(dim=0)  
        
        total_reward = avg_rewards.sum() 
        print(f'Episode {episode}, Total reward: {total_reward:.2f}')

        if hyper_paras['enable_early_stop'] and early_stopper.step(total_reward):
            print(f"Early stopping at episode {episode}")
            break

        if episode%5==1 or original_reward < total_reward: 
            fn = 'generative_model_ema' if args.ema_decay > 0 else 'generative_model'
            fn = f'{fn}_{episode}.npy'

            root_save_path = f"outputs_rl/tuned/{hyper_paras['overall_timestamp']}/{args.exp_name}"
            os.makedirs(root_save_path, exist_ok=True)

            utils.save_model(optimizer, f'{root_save_path}/optim.npy')
            utils.save_model(generative_model.module if hasattr(generative_model, "module") else generative_model, f'{root_save_path}/{fn}')
            original_reward = total_reward
            original_validity = validity
            print(f'Model saved at {root_save_path}/{fn}')

        for _ in range(hyper_paras['reuse']): 
            ppo_update(generative_model, optimizer, scheduler, latents_n, next_latents_n, log_probs_n, batch_n, advantages, reward_vector=reward_vector,
                       scalar_rewards=None, weights=weights, device=device, clip_param=hyper_paras['clip_param'], scaler=scaler, 
                       episode=episode, num_episodes=num_episodes, hyper_paras=hyper_paras, args=args)
            if episode == 0:
                init_reward_vector = reward_vector   
        
    return generative_model

def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend="nccl")  # Use NCCL backend for GPUs
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])  # Automatically assigned by torchrun
    torch.cuda.set_device(local_rank)
    return rank, local_rank

if __name__ == "__main__":

    from egnn import models
    import argparse

    rank, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    print(f"Process {rank} using GPU {local_rank}")


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--mode', type=str, default="baseline",
                        help='baseline, desired, None')
    parser.add_argument('--baseline', type=str, default="test",
                        help='train, test, valid')
    parser.add_argument('--timestamp', type=str, default="xx-xx-xx-overlap",
                        help='timestamp')
    parser.add_argument('--GPU_id', type=str, default="7",
                        help='gpu id')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=str2bool, default=False,
                        help='Should save samples to xyz files.')

    eval_args, unparsed_args = parser.parse_known_args()

    print(f'baseline type: {eval_args.baseline}')
    print(f'mode: {eval_args.mode}')
    print(f'GPU ID: {eval_args.GPU_id}')

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)


    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders[eval_args.baseline], mode=eval_args.mode, RL=True) # train

    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)


    try:
        generative_model.load_state_dict(flow_state_dict)
    except:
        new_state_dict = {}
        for k, v in flow_state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        flow_state_dict = new_state_dict
        generative_model.load_state_dict(flow_state_dict)

    generative_model.to(device)
    generative_model = torch.compile(generative_model)

    model = DDP(generative_model, device_ids=[local_rank])

    hyper_paras = {
        
        'save_model': True,
        'enable_blackbox_restore': True,
        'new_cutoff': None,  
        'property_list': ['qed', 'sas', 'affinity'],
        'valid_bonus': 1.20,
        'unique_bonus': 1.55,
        'novel_bonus': 1.60,
        'clip_param': 3e-4,
        'rl_lr': 1e-5,
        'reuse': 3, 
        'entropy_coef': 0.0,  
        'num_episodes': 300,  
        'enable_early_stop': False,


        'kl_coef': 0.00,  
        'ablation': 'uncertainty',  
        'uncertainty_method': 'pio', 
        'tasks': ['fitness', 'diversity'], 

        'top_flag': 'fitness',
        'num_top_mol': 5,

        'use_scheduler': False,
        'n_samples': 128, 
        'batch_size': 128, 
        't_indices': 100,  
        
        'enable_global_stat_tracker': False,
        'manual_weights': None, 
        'dynamic_weight': False,
        'adv_clip': 5,
        'diversity_k': 2,
        'affinity_max': 12,
        'GlobalStatTracker': 100, 

        'overall_timestamp': eval_args.timestamp,
    }


    print('RL hyper-parameters:', hyper_paras)

    num_episodes = hyper_paras['num_episodes']
    n_samples = hyper_paras['n_samples']
    batch_size = hyper_paras['batch_size']

    global_stat_tracker = GlobalStatTracker(buffer_size=hyper_paras['GlobalStatTracker']) if hyper_paras['enable_global_stat_tracker'] else None
    if global_stat_tracker == None: print('global_stat_tracker is None')
    train_ppo_(generative_model, n_samples, batch_size, dataset_info=dataset_info, num_episodes=num_episodes, device=device, stat_tracker=global_stat_tracker, hyper_paras=hyper_paras, args=args)


    dist.destroy_process_group()



