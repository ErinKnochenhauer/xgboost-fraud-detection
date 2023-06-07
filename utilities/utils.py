# Functions for fraud detection 
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

def convert_ccy(amount, original_ccy):
    if original_ccy == 'GBP':
        converted_amount = amount * 1.14
    elif original_ccy == 'SGD':
        converted_amount = amount * 0.71
    elif original_ccy == 'EUR':
        converted_amount = amount * 1 
    elif  original_ccy == 'CNY':
        converted_amount = amount * 0.14
    elif original_ccy == 'JPY':
        converted_amount = amount * 0.0068
    elif original_ccy == 'HKD': 
        converted_amount = amount * 0.13
    elif original_ccy == 'BRL':
        converted_amount = amount * 0.2
    elif original_ccy == 'INR':
        converted_amount = amount * 0.012
    else:
        return amount
    return converted_amount


def prep_data(df):
    """
    Convert the DataFrame into two variable
    X: feature columns
    y: label column
    """
    X = df.loc[:, df.columns != 'fraud_flag'].values
    y = df.fraud_flag.values
    return X, y


def plot_correlation_matrix(corr_matrix):
    """Plot a pre-calculated correlation matrix"""
    pio.templates.default = "plotly_white"
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    heat = go.Heatmap(
        z=corr_matrix.mask(mask),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=px.colors.diverging.RdBu,
        zmin=-1,
        zmax=1
    )

    layout = go.Layout(
        title_text='Correlation Matrix', 
        title_x=0.5, 
        width=800, 
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig=go.Figure(data=[heat], layout=layout)
    fig.show()


def get_shap_values(X, model):
    """Returns shap values for independent variables"""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values


def get_best_n_estimators(X, y, min_estimators=50, max_estimators=300, steps=50):
    """
    Get the best number of estimators by minimizing the log loss of 
    a range of values.
    
    """
    model = XGBClassifier()
    n_estimators = range(min_estimators, max_estimators, steps)
    param_grid = dict(n_estimators=n_estimators)
    kfold = StratifiedKFold(
        n_splits=10, 
        shuffle=True, 
        random_state=7
    )
    grid_search = GridSearchCV(
        model,
        param_grid, 
        scoring="neg_log_loss", 
        n_jobs=-1, 
        cv=kfold
    )
    grid_result = grid_search.fit(X, y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result.best_params_['n_estimators']
