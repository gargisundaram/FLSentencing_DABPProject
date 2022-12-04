# FLSentencing_DABPProject

Main Cleaning Files:
 - Dropped all crime data
 - Create CC Master
     - Join with President, House, Senate, Circuit-County Crosswalk, Normalized Population Data
     - Output: Circuit County Master with circuit-county-year level data about political environment
     - Dropped all crime data because did not exist for 2021 
 - Clean SAO (takes 2 files)
     - Specific Actions
 - Clean FDOC (takes 2 files)
     - Offenses and Root (demographic characteristics) for active and released inmates
     - Used prior prison offenses, current prison sentece stacked for active and released inmates
     - Results in set of offenses by person, date adjudicated time description, count of prior times, demographics
     - kept smallest prison term when given options

Create Model DF (1 file, CC Master, 2 args): 
 - Combine Clean FDOC, Clean SAO, Create CC Master
     - results in dummy variables, no NAs, no categorical columns
     - run it 4 times to get 4 model datasets: drug offenses, theft offenses, drug specific action, theft specific action

Model files (1 df only)
 - CART Models
 - Random Forest Model
 - XGBoost Model

