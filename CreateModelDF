def model_df(datafile, obstype, crimetype):
  
  df = pd.read_csv(datafile)
  ccm = pd.read_csv('circuit_county_year_master.csv')

  df.columns = df.columns.str.upper()

  if obstype == 'offense':

    if crimetype == 'drug':

      drug_dict = {'POSSESION': ['POS'],
                   'MARIJUANA':['MARIJUANA', 'CANN', 'SYN CAN'], 
                   'COCAINE':['COCAINE', 'COC'], 
                   'PRESCRIPTION': ['HYDROC', 'OXY', 'OP.ILL.PAIN MGT'],
                   'ILL_OPIOID' : ['FENTANYL', 'HEROIN','HER\.', 'HER,', 'HER/', 'OPIUM'],
                   'METH' : ['METH', 'AMPH'],
                   'HALL' : ['HALLUCINOGEN', 'LSD', 'PCP'],
                   'DATE_RAPE' : ['ROHYPNL', 'GBL', 'GHB'],
                   'BARBITURATES': ['BARB'],
                   'BATHSALTS':['BATHSALT'],
                   'MDMA':['MDMA']}
      
      for code in drug_dict.keys():
        df[code] = df['ADJUDICATIONCHARGE_DESCR'].str.contains('|'.join(drug_dict[code])).astype(int)

    if crimetype == 'theft':

      theft_dict = {
          'BURGLARY' : ['BURG', 'BURGLARY'],
          'GRAND_THEFT' : ['GR.', 'GRAND'],
          'HOME_INVASION' : ['HOME'],
          'PETIT_THEFT' : ['PETIT'],
          'RETAIL_THEFT' : ['RETAIL'],
          'ROBBERY' : ['ROBB'],
          'PROPERTY_THEFT' : ['PROP'],
          'SNATCH' : ['SNATCH'],
          'ELDER_ABUSE' : ['65YO'],
          'GOVT_THEFT' : ['STATE'],
          'SUBSTANCE_THEFT' : ['SUBSTANCE'],
          'DEADLYWEAPON' : ['DLY.WPN', 'DEADLY WPN', 'DW'],
          'NOWEAPON' : ['NO GUN/DDLY.WPN', 'NO WEAPON'],
          'NONDEADLYWEAPON' : ['OR WEAPO', 'OTHER WPN', 'WPN-NOT DEADLY']
      }

      for code in theft_dict.keys():
        df[code] = df['ADJUDICATIONCHARGE_DESCR'].str.contains('|'.join(theft_dict[code])).astype(int)


    df_dropcols = ['DCNUMBER','DATEADJUDICATED', 'ADJUDICATIONCHARGE_DESCR', 'RELEASEDATEFLAG_DESCR']
    df_dummycols = ['RACE', 'SEX']
    
  if obstype == 'action':
    df_dropcols = ['CASE_ID','CHARGE_ID','CASE_CREATED_DATE']
    df_dummycols = ['CHARGE_DEGREE','CHARGE_LEVEL', 'RACE', 'SEX', 'OFFENSE']
      
  df['COUNTY'] = df['COUNTY'].str.upper()
  df = pd.get_dummies(df, prefix=df_dummycols, columns=df_dummycols)
  df = df.drop(df_dropcols, axis = 1)

  final = pd.merge(df, ccm, how='left', on=['COUNTY','YEAR'])
  final_dummycols = ['COUNTY', 'YEAR', 'CIRCUIT', 'SA_NAME', 'POLITICAL_PARTY']
  final = pd.get_dummies(final, prefix=final_dummycols, columns=final_dummycols)

  final_filename = crimetype+'_'+obstype+'modeling_data.csv'
  final.to_csv(final_filename, index = False)

  return final
