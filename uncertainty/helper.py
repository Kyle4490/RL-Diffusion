import numpy as np
from scipy.stats import norm

def upper_confidence_bound(predictions, variances, kappa=1.0, minimize=False):
    """
    Calculate Upper Confidence Bound (UCB) fitness.
    
    Args:
        predictions (np.ndarray): Predicted mean values.
        variances (np.ndarray): Predicted variances.
        kappa (float): Exploration-exploitation trade-off parameter.
        minimize (bool): True if objective is minimization.
    
    Returns:
        np.ndarray: UCB fitness scores.
    """
    std_devs = np.sqrt(variances)
    ucb = predictions + kappa * std_devs
    if minimize:
        ucb = -ucb  # For minimization, negate to prefer smaller values
    return ucb

def mean_variance_combination(predictions, variances, beta=0.1, minimize=False):
    """
    Calculate Mean-Variance Combination (MVC) fitness.
    
    Args:
        predictions (np.ndarray): Predicted mean values.
        variances (np.ndarray): Predicted variances.
        beta (float): Penalty coefficient for variance.
        minimize (bool): True if objective is minimization.
    
    Returns:
        np.ndarray: MVC fitness scores.
    """
    mvc = predictions - beta * variances
    if minimize:
        mvc = -mvc  # For minimization, negate to prefer smaller values
    return mvc

def confidence_interval_weighted_width(predictions, variances, z=1.96, w_lower=0.5, w_upper=0.5, minimize=False):
    """
    Calculate Confidence Interval Weighted Width (CIWW) fitness.
    
    Args:
        predictions (np.ndarray): Predicted mean values.
        variances (np.ndarray): Predicted variances.
        z (float): Z-score for confidence interval (e.g., 1.96 for 95% CI).
        w_lower (float): Weight for lower confidence bound.
        w_upper (float): Weight for upper confidence bound.
        minimize (bool): True if objective is minimization.
    
    Returns:
        np.ndarray: CIWW fitness scores.
    """
    std_devs = np.sqrt(variances)
    lower_ci = predictions - z * std_devs
    upper_ci = predictions + z * std_devs
    ciww = w_lower * lower_ci + w_upper * upper_ci
    if minimize:
        ciww = -ciww  # For minimization, negate to prefer smaller values
    return ciww

def ucb(df, columns_list, dataset, property, objectives, kappa=1.0):
    """
    Calculate UCB fitness for a property.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions and variances.
        columns_list (list): Columns of the DataFrame (last two are prediction and variance).
        dataset (str): Dataset name (for cutoffs).
        property (str): Property name.
        objectives (dict): Objective directions.
        kappa (float): UCB parameter.
    
    Returns:
        np.ndarray: UCB fitness scores.
    """
    pred = df[columns_list[-2]].values
    var = df[columns_list[-1]].values + 1e-8  # Avoid zero variance
    var = np.where(var > 10000, 10000, var)  # Cap large variances
    objective = objectives[property]
    ucb_fitness = upper_confidence_bound(pred, var, kappa, minimize=(objective == "min"))
    return ucb_fitness

def mvc(df, columns_list, dataset, property, objectives, beta=0.1):
    """
    Calculate MVC fitness for a property.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions and variances.
        columns_list (list): Columns of the DataFrame (last two are prediction and variance).
        dataset (str): Dataset name (for cutoffs).
        property (str): Property name.
        objectives (dict): Objective directions.
        beta (float): MVC penalty coefficient.
    
    Returns:
        np.ndarray: MVC fitness scores.
    """
    pred = df[columns_list[-2]].values
    var = df[columns_list[-1]].values + 1e-8
    var = np.where(var > 10000, 10000, var)
    objective = objectives[property]
    mvc_fitness = mean_variance_combination(pred, var, beta, minimize=(objective == "min"))
    return mvc_fitness

def ciww(df, columns_list, dataset, property, objectives, z=1.96, w_lower=0.5, w_upper=0.5):
    """
    Calculate CIWW fitness for a property.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions and variances.
        columns_list (list): Columns of the DataFrame (last two are prediction and variance).
        dataset (str): Dataset name (for cutoffs).
        property (str): Property name.
        objectives (dict): Objective directions.
        z (float): Z-score for confidence interval.
        w_lower (float): Weight for lower CI.
        w_upper (float): Weight for upper CI.
    
    Returns:
        np.ndarray: CIWW fitness scores.
    """
    pred = df[columns_list[-2]].values
    var = df[columns_list[-1]].values + 1e-8
    var = np.where(var > 10000, 10000, var)
    objective = objectives[property]
    ciww_fitness = confidence_interval_weighted_width(pred, var, z, w_lower, w_upper, minimize=(objective == "min"))
    return ciww_fitness