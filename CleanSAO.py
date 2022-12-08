import numpy as np
import pandas as pd

def clean_sao(filename):
    dropcols = ['VICTIM_ID', "AGENCY_NAME", 'STATE', 'UCN_COUNTY_CODE',
            'CORRELATION_ID', 'UNIQUE_CORRELATION_ID', 'OWNER_ORI', 'DEFENDANT_ID', 'FINAL_ACTION_PROSECUTOR', 'DEF_STATUTE_CHAPTER_GROUPING',
            'VICTIM_AGE', 'VICTIM_RACE_TYPE_DESC', 'VICTIM_SEX_TYPE_DESC', 'VICTIM_ETHNICITY_TYPE_DESC', 'VICTIM_RELATIONSHIP_TO_OFFENDER_CODE',
            'DEF_PROSEC_PRETRIAL_DIVER_AGREE_DATE', 'DRUG_TYPE_DESC', 'DEF_STATUTE_CHAPTER_CODE', 'OFFENSE_CODE',
            'DEF_STATUTE_SECTION', 'DEF_STATUTE_SUBSECTION', 'DEFENDANT_AGE', 'DEFENDANT_ETHNICITY_TYPE_DESC', 'STATUTE']
    drug_keywords = ['Drug', 'Heroin', 'Cocaine', 'Marijuana', 'Opium', 'Amphetamine', 'Hallucinogen', 'Narcotic']
    theft_keywords = ['Robbery', 'Theft', 'Burglary', 'Stolen', 'Shoplift', 'Larceny']
    timecols = ['CASE_CREATED_DATE', 'FINAL_DECISION']
    
    df = pd.read_csv(filename)
    
    df.drop(dropcols, axis = 1, inplace = True)
    
    df.drop_duplicates(inplace = True)
    
    for col in timecols:
        df[col] = pd.to_datetime(df[col]).dt.date.astype('datetime64[ns]')

    df['YEAR'] = df.FINAL_DECISION.dt.year.astype('int')
    df.drop(['FINAL_DECISION'], axis=1, inplace = True)
    df = df[(df.YEAR > 2011) & (df.YEAR<2022)]

    df.rename(columns = {'COUNTY_DESCRIPTION' : "COUNTY",
                         'DEFENDANT_RACE_TYPE_DESC':'RACE',
                         'DEFENDANT_SEX_TYPE_DESC':'SEX',
                         'OFFENSE_FCIC_TYPE_DESC':'OFFENSE'}, inplace = True)

    drugsubset = np.where(df.OFFENSE_FCIC_TYPE_DESC.str.contains('|'.join(drug_keywords)))[0].tolist()
    theftsubset = np.where(df.OFFENSE_FCIC_TYPE_DESC.str.contains('|'.join(theft_keywords)))[0].tolist()

    drug_df = df.iloc[drugsubset].reset_index(drop = True)
    theft_df = df.iloc[theftsubset].reset_index(drop = True)

    drug_df.to_csv('drug_specific_actions_clean.csv', index = False)
    theft_df.to_csv('theft_specific_actions_clean.csv', index = False)
    
    return drug_df, theft_df
