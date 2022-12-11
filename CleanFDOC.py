import numpy as np
import pandas as pd

def clean_fdoc(filenames):
  #read-in file
  file_dict = {}
  for file in filenames:
    file_title = file[0:-4]
    df = pd.read_csv(file, low_memory=False)
    df['DCNumber'] = df.DCNumber.astype(str)
    df.drop(columns = ['Unnamed: 0'], inplace = True)
    file_dict[file_title] = df

  #concat offense files  
  offense_keys = ['Active_Offenses_PRPR_sample', 'Active_Offenses_CPS_sample', 'Release_Offenses_PRPR_sample', 'Release_Offenses_CPS_sample']
  offense_dfs = [file_dict[key] for key in offense_keys]
  df = pd.concat(offense_dfs)

  #format datetime columns
  timecols = ['DateAdjudicated', 'OffenseDate']
  for col in timecols:
      df[col] = df[col].str.replace('300', '200')
      df[col] = df[col].str.replace('299', '199')
      df[col] = pd.to_datetime(df[col]).dt.date.astype('datetime64')
  df.dropna(subset = ['DateAdjudicated'], inplace = True)

  #format county
  df['County'] = df.County.str.replace('SAINT', 'ST.')
  
  #get year and subset to daterange
  df['Year'] = df.DateAdjudicated.dt.year.astype(int)
  df = df[(df.Year > 2011) & (df.Year < 2022)].reset_index(drop = True)

  #create charge_count and priors features
  vars = ['DCNumber', 'DateAdjudicated', 'adjudicationcharge_descr', 'County', 'prisonterm', 'Year']
  df = df.groupby(vars, as_index=False).agg(charge_count = ('Sequence', 'count')).sort_values(['DCNumber', 'DateAdjudicated', 'adjudicationcharge_descr', 'prisonterm'])
  df = df.drop_duplicates(['DCNumber', 'DateAdjudicated', 'adjudicationcharge_descr'], keep = 'first')
  priors = df[['DCNumber', 'DateAdjudicated']].sort_values('DateAdjudicated').drop_duplicates()
  priors['prior_adj'] = priors.groupby('DCNumber').cumcount()
  df = df.merge(priors, how = 'left', on = ['DCNumber', 'DateAdjudicated'])

  #concat demographic files  
  demo_keys = ['Active_Root', 'Release_Root']
  demo_dfs = [file_dict[key] for key in demo_keys]
  demo = pd.concat(demo_dfs)

  ##clean demographic data
  #keep only necessary cols
  dem_cols = ['DCNumber', 'Race', 'Sex', 'BirthDate', 'releasedateflag_descr']
  demo = demo[dem_cols]
  #drop duplicates
  demo = demo.drop_duplicates(subset = 'DCNumber')
  #format birthdate as datetime
  demo['BirthDate'] = pd.to_datetime(demo.BirthDate).dt.date.astype('datetime64')

  #merge demographic data to offense data
  df = df.merge(demo, how = 'left', on = 'DCNumber')

  #calculate age at time of ajudication
  df['Age'] = np.floor((df.DateAdjudicated - df.BirthDate).dt.days/365.25).astype(int)

  #create target column
  df['prisonterm'] = df.prisonterm.astype(str).str.zfill(7)
  df['term_years'] = df.prisonterm.str[0:3].astype(int) + df.prisonterm.str[4:6].astype(int)/12 +  df.prisonterm.str[6:].astype(int)/365.25

  #drop any unnecessary columns
  dropcols = ['prisonterm', 'BirthDate']
  df = df.drop(dropcols, axis = 1)

  theft_keywords = ['THEFT', 'BURG', 'STOLEN', 'ROBB', 'SNATCH', 'GT ']
  drug_keywords = ['MARIJUANA', 'COCAINE', 'DRUG', 'GRAM', 'METH', 'COC', 'CLD', 'MFG', 'HYDROC', 'OP.ILL.PAIN MGT',
        'OXYCOD', 'LSD', 'BD', 'ROHYPNL', 'FENTANYL', 'SYN CAN', 'PCP', 'GBL', 'HER\.', 'HER,', 'HER/',
        'HEROIN', 'MDMA', 'OPIUM', 'CANN', 'PHEN', 'AMPH', 'DRG', 'GHB', 'BATHSALT', 'BARBITUATE', 'HALLUCINOGEN',
        'SUBSTANCE', 'CONTR SUB', 'CONT.SUB', 'CONTR.SUB', 'CONTROL.SUB', 'MAN SUB', 'CON.SUB', 'OBT.SUB']

  drug_rows = np.where(df.adjudicationcharge_descr.str.contains('|'.join(drug_keywords)))[0].tolist()
  theft_rows = np.where(df.adjudicationcharge_descr.str.contains('|'.join(theft_keywords)))[0].tolist()

  drug_df = df.iloc[drug_rows].reset_index(drop = True)
  theft_df = df.iloc[theft_rows].reset_index(drop = True)
  
  drug_df.to_csv('drug_offenses_data_clean.csv', index = False)  
  theft_df.to_csv('theft_offenses_data_clean.csv', index = False)

  return drug_df, theft_df
