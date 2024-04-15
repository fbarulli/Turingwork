
import pandas as pd
import numpy as np
import seaborn as sns
#import statsmodels.api as stats
from scipy import stats
from sklearn.model_selection import StratifiedKFold



from matplotlib import pyplot as plt



def hypothesis_test_all_qualities(df, alpha=0.01):
    """
    Perform a hypothesis test for each unique quality level in the dataset.

    Parameters:
    - df (DataFrame): The input DataFrame containing quality and alcohol data.
    - alpha (float): Significance level for hypothesis testing.

    Returns:
    - DataFrame: Results of hypothesis testing for each quality level.

    Example:
     results_df = hypothesis_test_all_qualities(df)
    """
    quality_values = df['quality'].unique()
    
    results = []

    for quality_value in quality_values:
        
        sample_mean = df[df['quality'] == quality_value]['alcohol'].mean()
        population_mean = df['alcohol'].mean()
        population_std = df['alcohol'].std()
        sample_size = len(df[df['quality'] == quality_value]['alcohol'])
        
        
        z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
        
        
        z_critical = stats.norm.ppf(1 - alpha)
        
        
        reject_null_critical = z_score > z_critical
        
        
        p_value = round(1 - stats.norm.cdf(z_score), 4)
        

        reject_null_p_value = p_value < alpha
        

        results.append({
            'Quality': quality_value,
            'Z-Score': z_score,
            'Critical Z-Score': z_critical,
            'Reject Null (Critical Z-Score)': reject_null_critical,
            'P-Value': p_value,
            'Reject Null (P-Value)': reject_null_p_value
        })

    results_df = pd.DataFrame(results)

    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Quality', y='Reject Null (P-Value)', data=results_df, palette='BuPu_r')
    plt.title('Hypothesis Test Results for Different Quality Levels')
    plt.xlabel('Quality Level')
    plt.ylabel('Reject Null Hypothesis (P-Value)')
    sns.despine()
    plt.show()
    

    return results_df







def create_folds(df, n_s=5, n_grp=None, task='classification'):
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
            df['grp'] = pd.cut(df.alcohol, n_grp, labels=False)
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