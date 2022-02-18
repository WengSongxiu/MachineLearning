import pandas as pd;
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["axes.unicode_minus"] = False
mpl.style.use('ggplot')

import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, make_scorer, f1_score, fbeta_score, precision_score, \
    roc_auc_score, accuracy_score, precision_recall_curve
from itertools import combinations
import xgboost as xgb
from scipy.stats import beta, norm


####################################################################################################
####################################################################################################
######################################                        ######################################
###################################### 10. FEATURE EVALUATION ######################################
######################################                        ######################################
####################################################################################################
####################################################################################################


def plot_hist(df, feat, bins=20, ax=None):
    """
    Plot histogram

    : params df: input dataframe
    : params feat: feature to be plot
    : params bins: number of bins
    """
    df = df.dropna(subset=[feat])
    (mu, sigma) = norm.fit(df[feat])
    # fit a normally distributed curve
    bins = min(min(bins, df[feat].nunique()) * 2, 100)
    if not ax:
        f, ax = plt.subplots(dpi=100)
    n, bins, patches = ax.hist(df[feat], bins, density=True, facecolor='orange', alpha=0.75)
    y = norm.pdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--', linewidth=2)
    plt.ylabel('Probability')
    plt.xlabel(feat)


def plot_hist_all(df, target, bins=20):
    """
    Plot histogram

    : params df: input dataframe
    : params feat: feature to be plot
    : params bins: number of bins
    """
    width = int(df.shape[1]) + 1
    fig_inx = 1
    fig = plt.figure(figsize=(30, 6 * width))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.3)

    col = [i for i in df.select_dtypes(exclude=[object]).columns if i not in set(target)]
    for i in col:
        ax = fig.add_subplot(width, 5, fig_inx)
        plot_hist(df, i, bins=bins, ax=ax)
        fig_inx += 1

