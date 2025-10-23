import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.stats import norm
import subprocess, argparse, os, shutil, json, sys, time
import pandas as pd
import numpy as np
from helper import ucb, mvc, ciww
# set a cutoff for each property  - done
# save smiles to csv  - done
# read smiles csv, pass into model, get results, save to a new csv  - done
# read all of data from csv  - done
# get mean and var for each mol  - done
# compute PIO for each mol  - done
# get overall MOPIO score  - done
# clean temp files  - done
import random
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


cutoffs = {
    'qm9':{  # Useful combination: 0.44, 7, -4.7 (0427091913)
        'qed' : 0.44, # 0.41, # 0.45, # 0.462, 0.41
        'sas' : 7, #7.35, # 6.5, # 7.357, 6.72
        'affinity' : -4.7, # -4.5, # -4.224, -4.73
    },
    'zinc15':{
        'qed' : 0.45, # 0.794,
        'sas' : 7, #7.326,
        'affinity' : -6.5, #-5.76, # -6.200,
    },
    'geom':{
        'qed' : 0.653,
        'sas' : 7.196,
        'affinity' : -6.730,
    },
    'pubchem':{
        'qed' : 0.651,
        'sas' : 7.183,
        'affinity' : -6.616,
    },
}

objectives = {
    'qed' : 'max',
    'sas' : 'min',
    'affinity' : 'min',
}

selected_timestamps = {
    'ensemble' : None,
    'evidential' : {
        'qm9' : {
            'qed': '2025-04-24-20-45-16',
            'sas': '2025-04-25-09-54-44',
            'affinity': '2025-04-24-21-44-58',
        },
        'zinc15' : {
            'qed': '2025-04-24-21-54-15',
            'sas': '2025-04-24-21-54-20',
            'affinity': '2025-04-24-21-54-43',
        },
        'pubchem' : {
            'qed': '2025-04-25-09-07-27',
            'sas': '2025-04-24-23-25-56',
            'affinity': '2025-04-24-23-26-17',
        },
        'geom' : {
            'qed': '2025-04-24-23-27-04',
            'sas': '2025-04-24-23-27-08',
            'affinity': '',
        },
    }
}

def set_cutoff(dataset, new_cutoff=None):
    """
    new_cutoff should be like:
        {
            'qed' : 0.651,
            'sas' : 7.183,
            'affinity' : -6.616,
        }
    """
    if new_cutoff != None:
        global cutoffs
        cutoffs[dataset] = new_cutoff

def calc_weighted_sum_fitness(self, smiles_list):
    preds, _ = self.predict(smiles_list)
    overall_fitness = 0
    for ii, target in enumerate(self.task_names):
        weight = self.target_weight_dict.get(target)
        objective = objectives[property]
        if objective == "maximize":
            overall_fitness += weight*preds[:, ii]
        elif objective == "minimize":
            overall_fitness += weight*preds[:, ii] * (-1)
    return overall_fitness

def expected_improvement(predictions, variances, cutoff, minimize=False):
    with np.errstate(divide='ignore', invalid='ignore'):
        if minimize:
            # For minimization, improvements are calculated as current best minus predictions
            improvements = cutoff - predictions
        else:
            # For maximization, improvements are predictions minus current best
            improvements = predictions - cutoff
        # Standard deviations
        std_devs = np.sqrt(variances)
        # Compute the Z value for the normal distribution
        Z = improvements / std_devs
        Z = np.where(std_devs > 0, Z, 0)  # Avoid division by zero
        # Calculate the EI
        ei = improvements * norm.cdf(Z) + std_devs * norm.pdf(Z)
        ei = np.where(std_devs > 0, ei, 0)  # EI is zero where std_dev is zero
    return ei

def ei(df, columns_list, cutoffs, dataset, property, objectives):
    pred = df[columns_list[-2]].values
    var = df[columns_list[-1]].values + 1e-8
    cutoff = cutoffs[dataset][property]
    objective = objectives[property]

    var = np.where(var > 10000, 10000, var)

    ei_fitness = expected_improvement(pred, var, cutoff, minimize=(objective == "min"))

    df["ei"] = ei_fitness
    prob_list = df["ei"].tolist()
    prob = np.array(prob_list)
    return prob
    

def gaussian_cdf(mean, variance, cutoff):
    std_dev = variance**0.5 
    dist = norm(mean, std_dev)  # construct the distribution
    cdf = dist.cdf(cutoff)  # probability that the distribution will be less than or equal to the cutoff
    return cdf

def gaussian_cdf_determined(mean, variance, cutoff, objective):
    cdf = gaussian_cdf(mean, variance, cutoff)
    if np.isnan(cdf):
        return 0
    
    if objective == 'max':
        return 1-cdf 
    elif objective == 'min':
        return cdf 
    else:
        return None

def compute_cutoffs(property_dict, quantile=50):
    high_good = {'qed': True, 'sas': False, 'affinity': False}
    
    cutoff_dict = {}
    for prop, values in property_dict.items():

        values = np.array(values)
        values = values[np.isfinite(values)]
        
        if len(values) == 0:
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

# def pio(df, columns_list, cutoffs, dataset, property, objectives, property_val_list=None):
#     df["cdf"] = df.apply(lambda row: gaussian_cdf_determined(
#                                 row[columns_list[-2]], 
#                                 row[columns_list[-1]], 
#                                 cutoffs[dataset][property], 
#                                 objective=objectives[property]), 
#                                 axis=1   #########
#                             )

#     prob_list = df["cdf"].tolist()
#     prob = np.array(prob_list)
#     return prob

def pio(df, columns_list, cutoffs, dataset, property, objectives, property_val_list=None, prob_thresh=0.8):

    cdf_list = []
    mean_col = columns_list[-2]
    var_col = columns_list[-1]
    cutoff = cutoffs[dataset][property]
    objective = objectives[property]

    for i, row in df.iterrows():
        mean = row[mean_col]
        var = row[var_col]
        # std = np.sqrt(var)
        cdf = gaussian_cdf_determined(mean, var, cutoff, objective)

        if property_val_list is not None: 
            true_y = property_val_list[i]
            if true_y == -float('inf'):
                cdf = 0.0
            elif cdf > prob_thresh:
                if (objective == "max" and true_y < cutoff) or (objective == "min" and true_y > cutoff):
                    cdf *= 0.1
            elif cdf < (1 - prob_thresh):
                if (objective == "max" and true_y >= cutoff) or (objective == "min" and true_y <= cutoff):
                    cdf = min(1.0, cdf * 10) if dataset=='qm9' else 1

        cdf_list.append(cdf)

    return np.array(cdf_list)
    
def get_pio_score(valid_smile_list, dataset, property_list, mode='evidential', timestamp=None, fitness='pio', new_cutoff=None, clean_temp_files=False, property_lists=None) -> list : 

    assert timestamp is not None, "timestamp cannot be none!"
    
    # ------ set dynamic cutoff ------
    # cutoffs
    if new_cutoff == None:
        new_cutoff = compute_cutoffs(property_lists, quantile=50)
    set_cutoff(dataset, new_cutoff=new_cutoff)
    # ------ set dynamic cutoff end ------

    fitness_list = []
    for property in property_list:

        model_timestamp = selected_timestamps[mode][dataset][property]
        model_dir = f"/data/lab_ph/kyle/projects/DrugDesign/uncertainty/model/{dataset}/{mode}/{property}_{model_timestamp}"

        generated_path = f'/data/lab_ph/kyle/projects/DrugDesign/uncertainty/generated/{timestamp}'
        os.makedirs(generated_path, exist_ok=True)

        test_file_path = f'{generated_path}/generated_smiles.csv'  ## generated mol smile file
        df_smile = pd.DataFrame(valid_smile_list, columns=["smiles"])
        df_smile.to_csv(test_file_path, index=False)

        pred_output_path = f"{generated_path}/{property}.csv"

        predict_cmd = [
            "chemprop_predict",
            "--checkpoint_dir", model_dir,
            "--test_path", test_file_path,
            "--preds_path", pred_output_path,
            "--uncertainty_method", "evidential_total",
            # "--seed", "42",
        ]

        try:
            subprocess.run(predict_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # create result csv
        except subprocess.CalledProcessError as e:
            print(f"Chemprop prediction failed: {e}")

        # --------------------------------------------------------------------------------
        for _ in range(10):
            if os.path.exists(pred_output_path):
                break
            time.sleep(0.5)
        
        if not os.path.exists(pred_output_path):
            try:
                subprocess.run(predict_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                df = pd.read_csv(pred_output_path)
            except:
                df = pd.DataFrame({
                    "smiles": valid_smile_list,
                    property: [0] * len(valid_smile_list),
                    f"{property}_evidential_total_uncal_var": [0] * len(valid_smile_list)
                })
        else:
            df = pd.read_csv(pred_output_path)
            # raise RuntimeError(f"{pred_output_path} not generated after waiting.")
        # --------------------------------------------------------------------------------

        # df = pd.read_csv(pred_output_path)
        
        df.replace("Invalid SMILES", 0, inplace=True)

        columns_list = df.columns.tolist()

        df[columns_list[-2]] = df[columns_list[-2]].astype(float)
        df[columns_list[-1]] = df[columns_list[-1]].astype(float)

        # print('cutoffs:', cutoffs)

        if fitness == 'pio':
            prob = pio(df, columns_list, cutoffs, dataset, property, objectives, property_val_list=property_lists[property])
            # -----------------------------
            prob = np.array(prob)
            if prob.size == 0 or np.any(np.isnan(prob)):
                prob = np.array([0.0])
            # -----------------------------
            fitness_list.append(prob)
        elif fitness == 'ei':
            prob = ei(df, columns_list, cutoffs, dataset, property, objectives)
            # -----------------------------
            prob = np.array(prob)
            if prob.size == 0 or np.any(np.isnan(prob)):
                prob = np.array([0.0])
            # -----------------------------
            fitness_list.append(prob)

        elif fitness == 'ucb':
            prob = ucb(df, columns_list, dataset, property, objectives, kappa=1.0)
            prob = np.array(prob)
            if prob.size == 0 or np.any(np.isnan(prob)):
                prob = np.array([0.0])
            fitness_list.append(prob)
        elif fitness == 'mvc':
            prob = mvc(df, columns_list, dataset, property, objectives, beta=0.1)
            prob = np.array(prob)
            if prob.size == 0 or np.any(np.isnan(prob)):
                prob = np.array([0.0])
            fitness_list.append(prob)
        elif fitness == 'ciww':
            prob = ciww(df, columns_list, dataset, property, objectives, z=1.96, w_lower=0.5, w_upper=0.5)
            prob = np.array(prob)
            if prob.size == 0 or np.any(np.isnan(prob)):
                prob = np.array([0.0])
            fitness_list.append(prob)
        else:
            raise ValueError("Unsupported fitness method. Use 'pio', 'ei', 'ucb', 'mvc', or 'ciww'.")
        
    if fitness_list != []:
        # -----------------------------
        shapes = [np.array(f).shape for f in fitness_list]
        max_len = max((s[0] if len(s) > 0 else 1) for s in shapes)

        fixed_fitness_list = []
        for f in fitness_list:
            f = np.array(f)
            if f.size == 0:
                f = np.array([0.0] * max_len)
            elif f.size < max_len:
                f = np.pad(f, (0, max_len - f.size), constant_values=0.0)
            fixed_fitness_list.append(f)
        # -----------------------------
        multiobjective_fitness = np.array(fitness_list)
        overall_fitness = np.prod(multiobjective_fitness, axis=0).tolist()
    else:
        overall_fitness = [0.0] * max_len

    if clean_temp_files:
        if os.path.exists(generated_path) and os.path.isdir(generated_path):
            shutil.rmtree(generated_path)

    # return overall_fitness
    return {
        "overall_fitness": overall_fitness,
        "new_cutoff": new_cutoff
    }




if __name__ == "__main__":
    # dataset = 'qm9'
    # property_list = ['qed', 'sas', 'affinity']
    # timestamp = 'AAAA'

    # valid_smile_list = [
    #     "O=Cn1nc(O)c(O)n1",
    #     "[N-]=c1[c-]c([O-])n[o+][n+]#[n+]1",
    #     "[C-2]=C1C(=O)[N+]#[N+]N=C1F",
    #     "C1#CC1=C1N=NN=N1",
    #     "C1=C=C(C2=[N+]=[N+]=N[N-]2)[C-]=1",
    #     "ABCDEF",
    #     "Cc1cnn(C[C@H](C)NCc2c(C)nn(C)c2Cl)C1",
    #     "BCDFS",
    #     "CCn1ncc2c1CCC[C@H]2N[C@H](C)C(=O)N1CCOCC1",
    #     "Cn1ccnc1CN1CCC[C@@H](CCc2cccc(F)c2)C1",
    #     "CC(C)N(Cc1cnn(C(C)(C)C)c1)C[C@@H]1CCC(=O)N1"
    # ]

    # for dataset in ['qm9', 'zinc15', 'pubchem']:
    #     scores = get_pio_score(valid_smile_list, dataset, property_list, mode='evidential', timestamp=timestamp, fitness='pio', new_cutoff=None, clean_temp_files=True)
    #     print(dataset, scores)


    args = json.loads(sys.argv[1])
    result = get_pio_score(**args)
    # print(result)
    print(json.dumps(result))


























        # print
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.max_rows", None)
        # print(df[["smiles", columns_list[-2], "cdf"]])
        # print(df["cdf"].tolist())