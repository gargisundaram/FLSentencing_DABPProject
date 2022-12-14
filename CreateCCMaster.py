import numpy as np
import pandas as pd
from functools import reduce

def create_ccm(safile, housefile, senatefile, presfile, circuitcountyfile) :
  sa = pd.read_csv(safile)
  ho = pd.read_csv(housefile)
  sen = pd.read_csv(senatefile)
  pr = pd.read_csv(presfile)
  cc = pd.read_csv(circuitcountyfile)

  dropcols = ['FULL_COUNTY', 'FULL COUNTY','STATE', 'STATE_PO','COUNTY_FIPS', 'OFFICE',
       'CANDIDATE', 'PARTY', 'CANDIDATEVOTES', 'TOTALVOTES']
  
  for df in [sa, ho, sen, pr, cc]:
    df.rename(columns = str.upper, inplace = True)
    df.rename(columns = {'PCT_DEM':'PCT_DEM_HOUSE', 'PERCENT DEMOCRAT': 'PCT_DEM_SENATE',
               'COUNTY_NAME':'COUNTY', '%_VOTES':'PCT_DEM_PRES'}, 
              inplace = True)
    
    df.dropna(inplace = True)
    
    for col in dropcols:
      if col in df.columns:
        df.drop(col, axis = 1, inplace = True)

    county_recode = {"OSECOLA":"OSCEOLA", "SAINT":'ST.', '-':''}
    if 'COUNTY' in df.columns:
      df['COUNTY'] = df['COUNTY'].str.upper()
      df['COUNTY'] = df['COUNTY'].replace(county_recode, regex = True)
      print()
    
    if 'YEAR' in df.columns:
      df['YEAR'] = df['YEAR'].astype(int)

  ccm = reduce(lambda  left,right: pd.merge(left,right,on=['COUNTY', 'YEAR'], how='left'), [ho, sen, pr]).merge(cc, how='left', on = 'COUNTY').merge(sa, how = 'left', on = ['CIRCUIT', 'YEAR'])
  ccm = ccm[ccm.YEAR > 2011]

  ccm.to_csv('circuit_county_year_master.csv')

  return ccm  
