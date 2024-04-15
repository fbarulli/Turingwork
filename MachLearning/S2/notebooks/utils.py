from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve


from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt


def create_folds(df, fold_length, n_grp=None, task='classification'):
    """
    Create folds for cross-validation in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - fold_length (int): Length of each fold.
    - n_grp (int, optional): Number of groups for classification task (if applicable).
    - task (str, optional): Type of task, either 'classification' or 'regression' (default is 'classification').

    Returns:
    - pd.DataFrame: Modified DataFrame with a new column 'Fold' indicating the assigned fold for each row.

    Raises:
    - ValueError: If an invalid task type is provided.
    """

    df['Fold'] = -1

    if task == 'classification':
        if n_grp is None:
            kf = KFold(n_splits=len(df) // fold_length, shuffle=True, random_state=42)
            target = df.stroke
        else:
            kf = StratifiedKFold(n_splits=len(df) // fold_length, shuffle=True, random_state=42)
            df['grp'] = pd.cut(df.stroke, n_grp, labels=False)
            target = df.grp
    elif task == 'regression':
        kf = KFold(n_splits=len(df) // fold_length, shuffle=True, random_state=42)
        target = df.stroke
    else:
        raise ValueError("Invalid task type. Use 'classification' or 'regression'")

    df.reset_index(drop=True, inplace=True)

    for fold_no, (t, v) in enumerate(kf.split(target, target)):
        df.loc[v, 'Fold'] = fold_no

    last_fold_length = fold_length if len(df) % fold_length == 0 else len(df) % fold_length
    df.iloc[-last_fold_length:, -1] = fold_no + 1
    
    return df


def bootstrap_sample(data1, data2, statistic, n_bootstrap=1000):
    combined_data = np.concatenate([data1, data2])
    n1 = len(data1)
    n2 = len(data2)
    bootstrap_statistics = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(combined_data, size=len(combined_data), replace=True)
        bootstrap_statistic = statistic(bootstrap_sample[:n1], bootstrap_sample[n1:])
        bootstrap_statistics.append(bootstrap_statistic)
    
    return bootstrap_statistics



def difference_in_means(data1, data2):
    return np.mean(data1) - np.mean(data2)










def preprocessing_pipeline(df):
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include='float64').columns
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('encoder', OneHotEncoder(sparse=False))
    ])
    
    model_transformer = ('model', 'passthrough') 
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
        model_transformer
    ])
    
    return preprocessor





def plot_comparison(col1, col2, data):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(x=col1, data=data, palette='mako', hue='stroke', ax=axes[0])
    axes[0].set_title(f'{col1.capitalize()} vs Stroke')
    axes[0].legend(['No Stroke', 'Stroke'])

    sns.histplot(x=col2, data=data, palette='viridis', hue='stroke', ax=axes[1])
    axes[1].set_title(f'{col2.capitalize()} vs Stroke')
    axes[1].legend(['No Stroke', 'Stroke'])

    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_comparison_2(col, df1, df2, palette1='mako', palette2='viridis'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    stroke_value1 = df1['stroke'].iloc[0]
    sns.histplot(x=col, data=df1, palette=palette1, hue='stroke', ax=axes[0])
    axes[0].set_title(f'{col.capitalize()} vs Stroke={stroke_value1}')

    stroke_value2 = df2['stroke'].iloc[0]
    sns.histplot(x=col, data=df2, palette=palette2, hue='stroke', ax=axes[1])
    axes[1].set_title(f'{col.capitalize()} vs Stroke={stroke_value2}')

    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_comparison_3(col, df1, df2, palette1='mako', palette2='viridis'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    unique_values1 = sorted(df1[col].unique())
    for value1 in unique_values1:
        df1_subset = df1[df1[col] == value1]
        sns.histplot(x=col, data=df1_subset, palette=palette1, hue='stroke', ax=axes[0])
        axes[0].set_title(f'{col} vs Stroke={df1["stroke"][0]}')


    unique_values2 = sorted(df2[col].unique())
    for value2 in unique_values2:
        df2_subset = df2[df2[col] == value2]
        sns.histplot(x=col, data=df2_subset, palette=palette2, hue='stroke', ax=axes[1])
        axes[1].set_title(f'{col} vs Stroke={df2["stroke"][0]}')


    plt.tight_layout()
    sns.despine()
    plt.show()






def preprocess_data(X):
    class CustomPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scale = MinMaxScaler()
            self.enc = LabelEncoder()
            self.num_cols = None
            self.cat_cols = None

        def fit(self, X, y=None):
            self.num_cols = X.select_dtypes(include='float64').columns
            self.cat_cols = X.select_dtypes(include='object').columns
            return self

        def transform(self, X):
            X_transformed = X.copy()

            imputer = SimpleImputer(strategy="mean")
            X_transformed['bmi'] = imputer.fit_transform(X_transformed[['bmi']])

            for col in self.cat_cols:
                X_transformed[col] = self.enc.fit_transform(X_transformed[col])

            for col in self.num_cols:
                X_transformed[col] = self.scale.fit_transform(X_transformed[[col]])

            return X_transformed

    preprocessor = CustomPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed




















'''
https://letsdatascience.com
http://karlrosaen.com/ml/learning-log/2016-06-20/
https://www.kaggle.com/code/ryanholbrook/principal-component-analysis
https://www.kaggle.com/code/ryanholbrook/mutual-information
https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
https://github.com/codiply/blog-ipython-notebooks/blob/master/scikit-learn-estimator-selection-helper.ipynb
https://www.kaggle.com/code/ryanholbrook/what-is-feature-engineering
https://www.mlebook.com/wiki/doku.php



'''