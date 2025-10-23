import os, torch
import subprocess
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import norm, pearsonr
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.cm as cm


from scipy import stats
from multiprocessing import Pool, cpu_count

def is_directory_empty(path):
    return os.path.isdir(path) and len(os.listdir(path)) == 0


def plot_parity(property_name=None, mode=None, data_dir=None, output_dir=None, pred_output_path=None, args=None):
    true_csv = os.path.join(data_dir, f"{property_name}_test.csv")
    true_df  = pd.read_csv(true_csv)      # ['smiles', property_name]
    
    pred_df  = pd.read_csv(pred_output_path)  #  ['smiles', f'{property_name}_prediction', f'{property_name}_{mode}_uncal_var']
    rename_dict = {
        property_name: f"{property_name}_prediction",
    }
    pred_df = pred_df.rename(columns=rename_dict)

    # 
    df = true_df.merge(pred_df, on="smiles")

    # 
    y_true = df[property_name]

    columns_list = df.columns.tolist()
    pred_col = columns_list[-2] #f"{property_name}_prediction"
    y_pred = df[pred_col]
    unc_col = columns_list[-1]  # f"{property_name}_{mode}_uncal_var"
    y_unc  = df[unc_col]

    # 3)  R²
    r2 = r2_score(y_true, y_pred)

    # 4) 
    plt.figure(figsize=(5,5), dpi=200)
    # 
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], '--', color='gray', linewidth=1)

    # 
    cmap = cm.viridis
    sc = plt.scatter(y_true, y_pred, c=y_unc, cmap=cmap, s=15, alpha=0.7, edgecolors='none')

    # 
    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label("Uncertainty variance", fontsize=12)

    # 
    plt.xlabel("True value", fontsize=14)
    plt.ylabel("Predicted value", fontsize=14)
    plt.title(f"{property_name.capitalize()} Parity Plot\n$R^2$ = {r2:.3f}", fontsize=16)
    plt.grid(alpha=0.3)

    # 
    out_png = os.path.join(output_dir, f"{property_name}_{mode}_parity_plot_{args.timestamp}.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)

    # plt.show()


def compute_interval(args):
    """
    Compute how many true values fall within the predicted confidence interval
    for a given confidence level.
    """
    confidence_level, unce, pred, true = args
    count = 0
    for u, p, t in zip(unce, pred, true):
        sigma = np.sqrt(u)
        lower, upper = stats.norm.interval(confidence_level, loc=p, scale=sigma)
        if lower <= t <= upper:
            count += 1
    return count

def compute_auce(pred, unce, true, q=20):
    # Define confidence levels
    conf_levels = [(i + 1) / q for i in range(q)]
    args = [(cl, unce, pred, true) for cl in conf_levels]
    num_procs = min(cpu_count(), q)
    with Pool(processes=num_procs) as pool:
        counts = pool.map(compute_interval, args)
    proportions = [counts[i] / len(true) for i in range(q)]
    # Compute AUCE
    auce = sum(abs(proportions[i] - conf_levels[i]) for i in range(q)) / q
    # Prepare plot data
    xs = [0.0] + conf_levels
    ys = [0.0] + proportions
    oracle = xs.copy()
    return auce, xs, ys, oracle

def plot_calibration_curve(
    data_dir,
    output_dir,
    pred_output_path,
    property_name,
    mode="mve",
    q=20,
    args=None,
):
    # File paths
    true_csv = os.path.join(data_dir, f"{property_name}_test.csv")
    # pred_csv = os.path.join(output_dir, "test_unc.csv")
    
    # Load data
    true_df = pd.read_csv(true_csv)  # Expect columns ['smiles', property_name]
    pred_df = pd.read_csv(pred_output_path)  # Expect ['smiles', 'prediction', 'uncertainty']
    rename_dict = {
        property_name: f"{property_name}_prediction",
    }
    pred_df = pred_df.rename(columns=rename_dict)
    
    # Merge on smiles
    df = true_df.merge(pred_df, on="smiles")
    y_true = df[property_name].values

    columns_list = df.columns.tolist()
    y_pred = df[columns_list[-2]].values  # f"{property_name}_prediction"
    y_unc  = df[columns_list[-1]].values  # f"{property_name}_{mode}_uncal_var"
    
    # Compute AUCE and calibration curve
    auce, xs, ys, oracle = compute_auce(y_pred, y_unc, y_true, q=q)
    
    # Plot
    plt.figure(figsize=(6, 6), dpi=200)
    plt.plot(xs, ys, color="darkorange", linewidth=2, label="Model")
    plt.plot(xs, oracle, color="gray", linestyle="--", linewidth=1.5, label="Ideal")
    plt.fill_between(xs, ys, oracle, color="lightgray", alpha=0.5)
    plt.xlabel("Confidence Level", fontsize=14)
    plt.ylabel("Proportion Within Interval", fontsize=14)
    plt.title(f"{property_name.capitalize()} Calibration Curve\nAUCE = {auce:.4f}", fontsize=16)
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    
    out_png = os.path.join(output_dir, f"{property_name}_{mode}_calibration_curve_{args.timestamp}.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    # plt.show()
    
    return auce

def split_and_save(data_list, prop_name, datasetName):

    path = '/data/lab_ph/kyle/projects/DrugDesign/uncertainty/data/'

    df = pd.DataFrame(data_list)
    Nmols = len(df)

    Ntrain = int(0.8 * Nmols)
    Ntest = int(0.1 * Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)
    train, valid, test, extra = np.split(data_perm, [Ntrain, Ntrain + Nvalid, Ntrain + Nvalid + Ntest])
    assert len(extra) == 0, f"Split mismatch: {len(train)}, {len(valid)}, {len(test)}, {len(extra)}"

    df.iloc[train].to_csv(f"{path}{datasetName}/{prop_name}_train.csv", index=False)
    df.iloc[valid].to_csv(f"{path}{datasetName}/{prop_name}_valid.csv", index=False)
    df.iloc[test].to_csv(f"{path}{datasetName}/{prop_name}_test.csv", index=False)

def get_datasets():
    datasets = ['qm9', 'zinc15', 'geom', 'pubchem']
    for data in datasets:
        path = f'/data/lab_ph/kyle/projects/DrugDesign/datasets/trainData_with_smiles/{data}_removeH_True.pt'
        molecules = torch.load(path)

        qed_data, sas_data, affinity_data = [], [], []

        for molecule in tqdm(molecules):
            smile = molecule['smiles']
            qed, sas, affinity = float(molecule['qed']), float(molecule['sas']), float(molecule['affinity'])

            qed_data.append({'smiles': smile, 'qed': qed})
            sas_data.append({'smiles': smile, 'sas': sas})
            affinity_data.append({'smiles': smile, 'affinity': affinity})

        split_and_save(qed_data, 'qed', data)
        split_and_save(sas_data, 'sas', data)
        split_and_save(affinity_data, 'affinity', data)
    print("Gotten dataset!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="qm9",
                        help='dataset name')
    parser.add_argument('--property', type=str, default="qed",
                        help='properties')
    parser.add_argument('--mode', type=str, default="ensemble",
                        help='train, test, valid')
    parser.add_argument('--timestamp', type=str, default="0000",
                        help='time')

    args, unparsed_args = parser.parse_known_args()

    dataset = args.dataset # 'qm9'
    property = args.property # 'qed'

    mode = 'evidential'  # args.mode # 'ensemble'

    print(f'dataset:{dataset}, property:{property}, mode:{mode}')

    data_dir = f'/data/lab_ph/kyle/projects/DrugDesign/uncertainty/data/{dataset}'
    if is_directory_empty(data_dir):
        print('getting data ...')
        get_datasets()

    model_dir = f"/data/lab_ph/kyle/projects/DrugDesign/uncertainty/model/{dataset}/{mode}/{property}_{args.timestamp}"
    output_dir = f'/data/lab_ph/kyle/projects/DrugDesign/uncertainty/outputs/{dataset}/{args.timestamp}'
    pred_output_path = f"{output_dir}/{mode}_{property}_{args.timestamp}.csv"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ================
    print("Step 1: Training Chemprop surrogate model with ensemble...")
    train_cmd = [
        "chemprop_train",
        "--data_path", os.path.join(data_dir, f"{property}_train.csv"),
        "--separate_val_path", os.path.join(data_dir, f"{property}_valid.csv"),
        "--separate_test_path", os.path.join(data_dir, f"{property}_test.csv"),

        "--dataset_type", "regression",
        '--warmup_epochs', '2',   
        '--num_workers', '20',
        '--grad_clip', '5',
        '--aggregation', 'sum',
        
        '--activation', 'PReLU',  # PReLU
        "--batch_size", "64",  # 64
        '--dropout', '0.1', # 0
        "--ensemble_size", "1",  # 10
        "--epochs", "40",
        "--evidential_regularization", "5e-3", # 5e-3
        "--ffn_hidden_size", "300",
        '--final_lr', '1e-5',
        '--hidden_size', '300',
        "--loss_function", "evidential", # "evidential",  # "mve",  # mve μ ± σ
        '--ffn_num_layers', '2',
        '--max_lr', '3e-3',
        '--init_lr', '1e-4',
        
        "--save_dir", model_dir,
        "--smiles_column", "smiles",
        "--target_columns", property, #"sas", "affinity",
        "--seed", "0",
        "--gpu", "0",  # 
        # "--evaluation_scores_path", metrics_path,
    ]
    print(train_cmd)
    subprocess.run(train_cmd, check=True)

    # ======== μ ± σ ========
    print("Step 2: Predicting properties and uncertainties for generated molecules...")
    predict_cmd = [
        "chemprop_predict",
        "--checkpoint_dir", model_dir,
        "--test_path", os.path.join(data_dir, f"{property}_test.csv"),
        "--preds_path", pred_output_path,
        "--uncertainty_method", "evidential_total", # mode, #"mve"
    ]

    subprocess.run(predict_cmd, check=True)

    print(f"Done! Predictions with uncertainty saved to: {pred_output_path}")

    plot_parity(property_name=property, mode=mode, data_dir=data_dir, output_dir=output_dir, pred_output_path=pred_output_path, args=args)

    plot_calibration_curve(
        data_dir, output_dir, pred_output_path, property_name=property, mode=mode, q=20, args=args)

    print('done!')

