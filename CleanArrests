def clean_arrests(file_name):
    df = pd.read_csv(file_name)
    
    df = df.drop(badrows)
    df = df.reset_index(drop=True)
    
    df.columns = cols
    
    df = df.replace(',', '', regex = True)
    df[cols[1:]] = df[cols[1:]].apply(pd.to_numeric)
    
    for col in df.columns[4:]:
        df = df.copy()
        newcol = col.replace('COUNT', 'RATE')
        df[newcol] = 100000*df[col]/df.POP
        df = df.drop(col, axis = 1)

    return df.copy()

def concat_arrests():
  cols = ['COUNTY', 'POP', 'ARREST_COUNT', 'ARREST_RATE'] 
  offense_types = ['ALLOFF', 'MURDER', 'MANSLAUGHTER', 'RAPE', 'ROBBERY', 'AGGASSAULT', 
                 'BURGLARY', 'LARCENY', 'MVTHEFT', 'KIDNAP', 'ARSON', 'SIMPASSAULT',
                'DRUG', 'BRIBERY', 'EMBEZZLEMENT', 'FRAUD', 'FORGERY', 'BLACKMAIL',
                'INTIMIDATION', 'PROSTITUTION', 'PART2SEX', 'STOLENPROP', 'DUI',
                'VANDAL', 'GAMBLE', 'WEAPONSVIOL', 'LIQUORVIOL', 'MISC']
  agecats = ['ADULT', 'JUV']
  gendercats = ['MALE', 'FEMALE']
  racecats = ['WHITE', 'BLACK', 'INDIAN', 'ASIAN']

  subcats = []
  for i in agecats:
    for j in gendercats:
        subcats.append(i+'_'+j)

  for k in racecats:
    subcats.append(k)
            
  for i in offense_types:
    for j in subcats:
        cols.append(i+'_'+j+'_ARREST_COUNT')

  #identify useless rows
  badrows = [0, 1, 69, 70, 71]

  arrest_df = pd.DataFrame()

  for year in range(2012, 2021):
    filename = 'arrests_'+str(year)+'.csv'
    df = clean_arrests(filename)
    df['YEAR'] = year
    arrest_df = pd.concat([arrest_df, df])
    arrest_df = arrest_df.reset_index(drop=True)

  arrest_df.to_csv('county_year_arrests_clean.csv', index = False)

  return arrest_df
