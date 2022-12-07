import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
import shap
from functools import reduce

#Read-in custom functions
from CleanSAO import clean_sao
from CleanFDOC import clean_fdoc
from CreateCCMaster import create_ccm
from CreateModelDF import model_df
from RunTrees import get_tree

#Clean charges(prosecutor specific actions) datasets (drug, theft)
drug_sa_clean, theft_sa_clean = clean_sao('CjdtSAOCase_00000.csv')

#Clean sentencing (offenses) datasets (drug, theft)
drug_offenses_clean, theft_offenses_clean = clean_fdoc(['Active_Offenses_PRPR.csv', 'Active_Offenses_CPS.csv', 
                                                       'Release_Offenses_PRPR.csv', 'Release_Offenses_CPS.csv', 
                                                       'Active_Root.csv', 'Release_Root.csv'])

#Circuit-county-year political/State Attorney dataset
ccm = create_ccm(safile = "SA_Political_Leanings.csv", 
                    housefile = "clean_house.csv", 
                    senatefile = "clean_senate.csv", 
                    presfile = "clean_pres.csv", 
                    circuitcountyfile="circuit_county_crosswalk.csv")

# Compile modeling datasets
drug_sa_df = model_df(drug_sa_clean, ccm, obstype = 'action', crimetype = 'drug')
theft_sa_df = model_df(theft_sa_clean, ccm, obstype = 'action', crimetype = 'theft')
drug_off_df = model_df(drug_offenses_clean, ccm, obstype = 'offense', crimetype = 'drug')
theft_off_df = model_df(theft_offenses_clean, ccm, obstype = 'offense', crimetype = 'theft')

# CART Models
cart = {'max_depth':[3, 6, 10], 
        'min_samples_split':[8, 20], 
        'min_samples_leaf': [10, 20, 100]}

# Classifiers
drug_sa_tree = get_tree(drug_sa_df, target = 'FINAL_ACTION_DESC', paramdict = cart, model = DecisionTreeClassifier,  seed = 42)
theft_sa_tree = get_tree(theft_sa_df, target = 'FINAL_ACTION_DESC', paramdict = cart, model = DecisionTreeClassifier, seed = 42)

# Regressors
drug_off_tree = get_tree(drug_off_df, target = 'TERM_YEARS', paramdict = cart, model = DecisionTreeRegressor, seed = 42)
theft_off_tree = get_tree(theft_off_df, 'TERM_YEARS', paramdict = cart, model = DecisionTreeRegressor, seed = 42)

#RF hyperarameter space
bag = {'max_depth':[10, 15], 
        'min_samples_leaf': [100, 200], 
        'max_features': [40, 80], 
        'n_estimators':[15, 20]} 

# RF Classifiers
drug_sa_tree = get_tree(drug_sa_df, target = 'FINAL_ACTION_DESC', paramdict = bag, model = RandomForestClassifier, seed = 42)
theft_sa_tree = get_tree(theft_sa_df, target = 'FINAL_ACTION_DESC', paramdict = bag, model = RandomForestClassifier, seed = 42)

# RF Regressors
drug_off_tree = get_tree(drug_off_df, target = 'TERM_YEARS', paramdict = bag, model = RandomForestRegressor, seed = 42)
theft_off_tree = get_tree(theft_off_df, 'TERM_YEARS', paramdict = bag, model = RandomForestRegressor, seed = 42)

# XGB hyperparameter space
bag = {'max_depth':[8, 12], 
        'min_samples_leaf': [5, 20],
        'max_features': [8, 18], 
        'n_estimators':[10, 40]}

# XGB Classifiers
drug_sa_tree = get_tree(drug_sa_df, target = 'FINAL_ACTION_DESC', paramdict = bag, model = XGBClassifier, seed =10)
theft_sa_tree = get_tree(theft_sa_df, target = 'FINAL_ACTION_DESC', paramdict = bag, model = XGBClassifier,seed=10)

drug_off_tree = get_tree(drug_off_df, target = 'TERM_YEARS', paramdict = bag, model = XGBRegressor, seed=42)
theft_off_tree = get_tree(theft_off_df, 'TERM_YEARS', paramdict = bag, model =XGBRegressor, seed = 42)