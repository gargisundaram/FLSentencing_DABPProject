import numpy as np
import pandas as pd

def model_df(df, ccm, obstype, crimetype):

  df.columns = df.columns.str.upper()
  df['COUNTY'] = df.COUNTY.str.upper()

  if obstype == 'offense':

    if crimetype == 'drug':

      drug_dict = {'POSSESSION': ['POS'],
                   'MARIJUANA':['MARIJUANA', 'CANN', 'SYN CAN'], 
                   'COCAINE':['COCAINE', 'COC'], 
                   'PRESCRIPTION_OPIOIDS': ['HYDROC', 'OXY', 'OP.ILL.PAIN MGT'],
                   'NON-PRESCRIPTION_OPIOIDS' : ['FENTANYL', 'HEROIN','HER\.', 'HER,', 'HER/', 'OPIUM'],
                   'METH' : ['METH', 'AMPH'],
                   'HALLUCINOGENS' : ['HALLUCINOGEN', 'LSD', 'PCP'],
                   'DATE_RAPE' : ['ROHYPNL', 'GBL', 'GHB'],
                   'BARBITURATES': ['BARB'],
                   'BATHSALTS':['BATHSALT'],
                   'MDMA':['MDMA']}
      
      for code in drug_dict.keys():
        df[code] = df['ADJUDICATIONCHARGE_DESCR'].str.contains('|'.join(drug_dict[code])).astype(int)

    if crimetype == 'theft':

      theft_dict = {
          'BURGLARY' : ['BURG', 'BURGLARY'],
          'GRAND THEFT' : ['GR.', 'GRAND'],
          'HOME_INVASION' : ['HOME'],
          'PETIT_THEFT' : ['PETIT'],
          'RETAIL_THEFT' : ['RETAIL'],
          'ROBBERY' : ['ROBB'],
          'PROPERTY_THEFT' : ['PROP'],
          'SNATCH' : ['SNATCH'],
          'ELDER_ABUSE' : ['65YO'],
          'GOVERNMENT_THEFT' : ['STATE'],
          'SUBSTANCE_THEFT' : ['SUBSTANCE'],
          'DEADLY_WEAPON' : ['DLY.WPN', 'DEADLY WPN', 'DW'],
          'NO_WEAPON' : ['NO GUN/DDLY.WPN', 'NO WEAPON'],
          'NON-DEADLY_WEAPON' : ['OR WEAPO', 'OTHER WPN', 'WPN-NOT DEADLY']
      }

      for code in theft_dict.keys():
        df[code] = df['ADJUDICATIONCHARGE_DESCR'].str.contains('|'.join(theft_dict[code])).astype(int)


    df = df.iloc[np.where(df['TERM_YEARS'] < 1000)]
    df.reset_index(inplace = True, drop = True)
    df_dropcols = ['DCNUMBER','DATEADJUDICATED', 'ADJUDICATIONCHARGE_DESCR', 'RELEASEDATEFLAG_DESCR']
    df_dummycols = ['RACE', 'SEX']
    
  if obstype == 'action':
    df_dropcols = ['CASE_ID','CHARGE_ID','CASE_CREATED_DATE']
    df_dummycols = ['CHARGE_DEGREE','CHARGE_LEVEL', 'RACE', 'SEX', 'OFFENSE']
    badcodes = ["Administratively Dismissed", "Transferred to Another Court", "Consolidated", "Pre-Trial Diversion"]
    df = df.drop(np.where(df['FINAL_ACTION_DESC'].str.contains('|'.join(badcodes)))[0], axis = 0)

    charge_code = {'Filed Pending Court':1,'Dropped or Abandoned':0,'No Action':0,'Nolle Prossed':0 }
    df = df.replace({"FINAL_ACTION_DESC": charge_code})
      
  df = pd.get_dummies(df, prefix=df_dummycols, columns=df_dummycols)
  df = df.drop(df_dropcols, axis = 1)

  final = pd.merge(df, ccm, how='left', on=['COUNTY','YEAR'])
  final_dummycols = ['COUNTY', 'YEAR', 'CIRCUIT', 'SA_NAME', 'POLITICAL_PARTY']
  final = pd.get_dummies(final, prefix=final_dummycols, columns=final_dummycols)
  final.dropna(inplace = True)

  final_filename = crimetype+'_'+obstype+'modeling_data.csv'
  final.to_csv(final_filename, index = False)

  return final
