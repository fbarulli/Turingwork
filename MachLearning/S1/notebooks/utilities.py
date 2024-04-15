from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
import seaborn as sns






def create_folds(df, n_s=2, n_grp=2, task='classification'):
    """
    Create folds for cross-validation in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - n_s (int, optional): Number of folds for cross-validation (default is 5).
    - n_grp (int, optional): Number of groups for classification task (if applicable).
    - task (str, optional): Type of task, either 'classification' or 'regression' (default is 'classification').

    Returns:
    - pd.DataFrame: Modified DataFrame with a new column 'Fold' indicating the assigned fold for each row.

    Raises:
    - ValueError: If an invalid task type is provided.

    Example:
     df = create_folds(df, n_s=5, n_grp=3, task='classification')
    """

    df['Fold'] = -1
    
    if task == 'classification':
        if n_grp is None:
            kf = KFold(n_splits=n_s, shuffle=True, random_state=42)
            target = df.alcohol
        else:
            kf = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=42)
            df['grp'] = pd.cut(df.AnnualIncome, n_grp, labels=False)
            target = df.grp
    elif task == 'regression':
        kf = KFold(n_splits=n_s, shuffle=True, random_state=42)
        target = df.alcohol
    else:
        raise ValueError("Invalid task type. Use 'classification' or 'regression'")
    
    df.reset_index(drop=True, inplace=True)  
    
    if task == 'classification':
        for fold_no, (t, v) in enumerate(kf.split(target, target)):
            df.loc[v, 'Fold'] = fold_no
    elif task == 'regression':
        for fold_no, (t, v) in enumerate(kf.split(df)):
            df.loc[v, 'Fold'] = fold_no
            
    return df