import torch
from torch.distributions.categorical import Categorical

import numpy as np
from egnn.models import EGNN_dynamics_QM9
import pandas as pd
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from scipy.stats import beta

def get_model(args, device, dataset_info, dataloader_train, mode='baseline'):
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

class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None, mode=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties

        # -------------------------------------------------------------
        size = len(dataloader.dataset.data[list(dataloader.dataset.data.keys())[0]])
        # self.conditions_tensor = {
        #     'qed' : torch.tensor(0.7 + 0.3 * beta.rvs(2, 5, size=size)),
        #     'sas' : torch.tensor(np.clip(4 * beta.rvs(5, 2, size=size), 0, 4)),
        #     'affinity' : torch.tensor(np.clip(np.random.normal(-9, 1, size=size), -np.inf, -7))
        # }
        # self.conditions_tensor = { 
        #     'qed' : torch.tensor(0.5 + 0.2 * beta.rvs(1.8, 4.0, size=size)),
        #     'sas' : torch.tensor(5.5 + 2.0 * beta.rvs(3.0, 2.5, size=size)),
        #     'affinity' : torch.tensor(-6 + 1.5 * beta.rvs(3.0, 4.5, size=size))
        # }
        # -------------------- sampling from sets -----------------------
        if mode != 'baseline':
            dataset_name = {13225:'qm9', 19862:'zinc15', 44428:'pubchem'}
            data_path = f'/data/lab_ph/kyle/projects/DrugDesign/uncertainty/data/merged/{dataset_name[size]}_merged.csv'
            qualified = get_dist(data_path, quantile = 50)[:size]
            self.conditions_tensor = {
                'qed': torch.tensor(qualified['qed'].tolist(), dtype=torch.float32),
                'sas': torch.tensor(qualified['sas'].tolist(), dtype=torch.float32),
                'affinity': torch.tensor(qualified['affinity'].tolist(), dtype=torch.float32)
            }
        # -------------------- sampling from sets end -----------------------

        # -------------------------------------------------------------

        for prop in properties:
            
            print(f'Baseline {prop}: {dataloader.dataset.data[prop].mean()}')
            # print(f'Desired: {prop}: {self.conditions_tensor[prop].mean()}')

            self.distributions[prop] = {}

            if mode == 'baseline':
                self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                    dataloader.dataset.data[prop],
                                    self.distributions[prop])
            else:
                print(f'Desired: {prop}: {self.conditions_tensor[prop].mean()}')
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
            # ------------------------ Fixed at Feb 24, 2025 start -----------------------
            # dist = self.distributions[prop][n_nodes]
            if n_nodes not in self.distributions[prop]:
                available_keys = list(self.distributions[prop].keys())
                closest_n = min(available_keys, key=lambda x: abs(x - n_nodes))
                print(f"Warning: node count {n_nodes} not found for {prop}. Using {closest_n} instead.")
                dist = self.distributions[prop][closest_n]
            else:
                dist = self.distributions[prop][n_nodes]
            # ------------------------ Fixed at Feb 24, 2025 end -----------------------
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


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
