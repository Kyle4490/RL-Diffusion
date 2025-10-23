# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils, re
import argparse
from qm9 import dataset
from qm9.models import get_model
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time, random
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses
from tqdm import tqdm
import numpy as np
import subprocess
from datetime import datetime
# from measureProperties import sdf2pdbqt


try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)

seeds = 2024
torch.manual_seed(seeds)
torch.cuda.manual_seed(seeds)
torch.cuda.manual_seed_all(seeds) 
np.random.seed(seeds)
random.seed(seeds)

def str2bool(value):
  if isinstance(value, bool):
    return value
  if value.lower() in {'true', 'yes', '1'}:
    return True
  elif value.lower() in {'false', 'no', '0'}:
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False, trial=0):
    
    # ----------------------- splitted codes ==> get middle first start-------------------------
    file_details = args.exp_name
    save_dir = f'{eval_args.model_path}/evaluate' #'outputs_middle'
    save_file = f'middle_{file_details}_trial_{trial}.pt'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(os.path.join(save_dir, save_file)):
        print('Testing without generation ...')
        molecules, dataset_info = torch.load(os.path.join(save_dir, save_file))
        stability_dict, rdkit_metrics = analyze_stability_for_molecules(molecules, dataset_info)
        return stability_dict, rdkit_metrics
    else:
        print("Save middle results (generated molecules) but don't test ...")
        batch_size = min(batch_size, n_samples)
        assert n_samples % batch_size == 0
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        start_time = time.time()
        print(f'model path: {eval_args.model_path}')
        for i in tqdm(range(int(n_samples/batch_size))):
            nodesxsample = nodes_dist.sample(batch_size)
            one_hot, charges, x, node_mask = sample(
                args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())

            current_num_samples = (i+1) * batch_size
            secs_per_sample = (time.time() - start_time) / current_num_samples
            print('\t %d/%d Molecules generated at %.2f secs/sample' % (
                current_num_samples, n_samples, secs_per_sample))

            if save_to_xyz:
                id_from = i * batch_size
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                qm9_visualizer.save_xyz_file(
                    join(eval_args.model_path, f'eval/analyzed_molecules_{timestamp}/'),
                    one_hot, charges, x, dataset_info, id_from, name='molecule',
                    node_mask=node_mask)

        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}

        torch.save([molecules, dataset_info], os.path.join(save_dir, save_file))
        print(f'Save middle results at: {os.path.join(save_dir, save_file)}')
        return None, None
    # ----------------------- get middle first end-------------------------

    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    print(f'model path: {eval_args.model_path}')
    for i in tqdm(range(int(n_samples/batch_size))):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, f'eval/analyzed_molecules_{timestamp}/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)

    return stability_dict, rdkit_metrics


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None

                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                        edge_mask, context)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def main(trial=0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--mode', type=str, default="baseline",
                        help='baseline, desired, None')
    parser.add_argument('--baseline', type=str, default="test",
                        help='train, test, valid')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=str2bool, default=False,
                        help='Should save samples to xyz files.')

    eval_args, unparsed_args = parser.parse_known_args()

    print(f'baseline type: {eval_args.baseline}')
    print(f'mode: {eval_args.mode}')
    print(f'model path: {eval_args.model_path}')

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders[eval_args.baseline], mode=eval_args.mode) # train

    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    # fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    # flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)

    # -------------------- auto matching
    try:
        fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    except:
        if args.ema_decay > 0:
            pattern = re.compile(r'generative_model_ema_(\d+)\.npy')
            candidates = []
            for f in os.listdir(eval_args.model_path):
                match = pattern.match(f)
                if match:
                    step = int(match.group(1))
                    candidates.append((step, f))
            
            if candidates:
                _, fn = max(candidates, key=lambda x: x[0])
            # else:
            #     fn = 'generative_model_ema.npy'
        else:
            pattern = re.compile(r'generative_model_(\d+)\.npy')
            candidates = []
            for f in os.listdir(eval_args.model_path):
                match = pattern.match(f)
                if match:
                    step = int(match.group(1))
                    candidates.append((step, f))
            
            if candidates:
                _, fn = max(candidates, key=lambda x: x[0])
            # else:
            #     fn = 'generative_model.npy'
        flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    # -------------------- auto matching end

    print(f'fn: {fn}')
    # ------------------------------------------------------
    try:
        generative_model.load_state_dict(flow_state_dict)
    except:
        new_state_dict = {}
        for k, v in flow_state_dict.items():
            if k.startswith('_orig_mod.'):
                k = k[len('_orig_mod.'):]
            if k.startswith('module.'):
                k = k[len('module.'):]
            # if k.startswith('module.'):
            #     new_state_dict[k[7:]] = v 
            # else:
            #     new_state_dict[k] = v
            new_state_dict[k] = v
        flow_state_dict = new_state_dict
        generative_model.load_state_dict(flow_state_dict)
    # ------------------------------------------------------
    print(f"Number of generated molecules: {eval_args.n_samples}")
    # Analyze stability, validity, uniqueness and novelty
    stability_dict, rdkit_metrics = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz, trial=trial)
    print(stability_dict)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
        print(f"qed: {rdkit_metrics[3]}, sas: {rdkit_metrics[4]}, affinity: {rdkit_metrics[5]}")
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")
        return None
    
    return {'validity' : rdkit_metrics[0], 
            'uniqueness' : rdkit_metrics[1], 
            'novelty' : rdkit_metrics[2],
            'atm_stable' : stability_dict['atm_stable'], 
            'mol_stable' : stability_dict['mol_stable'], 
            'qed' : rdkit_metrics[3],
            'sas' : rdkit_metrics[4],
            'affinity' : rdkit_metrics[5]
        }

    # # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    # if args.dataset == 'geom':
    #     val_name = 'val'
    #     num_passes = 1
    # else:
    #     val_name = 'valid'
    #     num_passes = 5

    # # Evaluate negative log-likelihood for the validation and test partitions
    # val_nll = test(args, generative_model, nodes_dist, device, dtype,
    #                dataloaders[val_name],
    #                partition='Val')
    # print(f'Final val nll {val_nll}')
    # test_nll = test(args, generative_model, nodes_dist, device, dtype,
    #                 dataloaders['test'],
    #                 partition='Test', num_passes=num_passes)
    # print(f'Final test nll {test_nll}')

    # print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    # with open(join(eval_args.model_path, 'eval_log.txt'), 'w') as f:
    #     print(f'Overview: val nll {val_nll} test nll {test_nll}',
    #           stability_dict,
    #           file=f)


if __name__ == "__main__":
    trials = 3
    metrics = {'validity' : [], 
                'uniqueness' : [], 
                'novelty' : [], 
                'atm_stable' : [], 
                'mol_stable' : [],
                'qed' : [],
                'sas' : [],
                'affinity' : []
            }
    # for i in range(trials):
    #     metric = main(i)
    #     if metric == None:
    #         continue
    #     for key in metrics.keys():
    #         metrics[key].append(metric[key])

    # -------------------------------------------
    print('first generating...')
    for i in range(trials):
        metric = main(i)
    
    print('start testing...')
    for i in range(trials):
        metric = main(i)
        if metric == None:
            continue
        for key in metrics.keys():
            metrics[key].append(metric[key])
    # -------------------------------------------

    print(metrics)
    print(f'\nNumber of Trials: {trials}')
    for key, data in metrics.items():
        mean, ci_range = np.mean(data), 1.96 * np.std(data, ddof=1) / np.sqrt(len(data))
        if key in ['qed', 'sas', 'affinity']:
            print(f"{key}: {mean:.2f} ± {ci_range:.2f}")
        else:
            print(f"{key}: {mean*100:.2f} ± {ci_range*100:.2f}")






    # import pybel
    # from openbabel import pybel
    # convert molecular xyz to sdf 
    # xyz_path = ''  

    # input_xyz = "/data/lab_ph/kyle/projects/DrugDesign/baselines/e3_diffusion_for_molecules/outputs/exp_qm9_without_h_conditional/eval/analyzed_molecules/molecule_000.xyz"
    # output_qdbqt = "/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/temp_data/molecule_000.pdbqt"
    
    # molecule = next(pybel.readfile("xyz", input_xyz))
    # molecule.write("sdf", output_sdf)

    # command = ['obabel', input_xyz, '-O', output_qdbqt, '--gen3d', '--partialcharge', 'gasteiger']
    # subprocess.run(command, capture_output=True, text=True)

    

    # # convert sdf to pdbqt xxxxx
    # molecule_file = 'sdf file'
    # output_file_molecule = 'pdbqt file'
    # sdf2pdbqt(molecule_file, output_file_molecule)

    # molecule_pdbqt_path = '/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation'



    # # below can work
    # # docking -> get docking results
    # working_directory = "/data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1"  
    # config_file = '/data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1/configs/evaluation_config.txt'
    # command = f"./QuickVina2-GPU-2-1 --config {config_file}"  
    # subprocess.run(command, shell=True, check=True, cwd=working_directory)
    
    # # collect docking scores
    # output_directory = '/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation_out'
    # all_processed_pdbqt = os.listdir(output_directory)
    # affinity = []
    # for pdbqt in all_processed_pdbqt:
    #     if pdbqt.endswith('.pdbqt'):
    #         pdbqt_fullPath = os.path.join(output_directory, pdbqt)
    #         with open(pdbqt_fullPath, 'r') as f:
    #             _ = f.readline().strip()
    #             affinity.append(float(f.readline().strip().split()[3]))
    
    # print(affinity)
    # print(np.mean(affinity))
    
    