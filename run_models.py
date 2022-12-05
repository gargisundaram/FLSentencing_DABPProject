import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier, plot_importance
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret import show
from interpret.blackbox import ShapKernel
from functools import reduce
from interpret import set_visualize_provider, show
from interpret.provider import InlineProvider
from interpret.blackbox import ShapKernel


from CleanSAO import clean_sao
from CleanFDOC import clean_fdoc
from CreateCCMaster import create_ccm
from CreateModelDF import model_df
from RunTrees import get_tree
