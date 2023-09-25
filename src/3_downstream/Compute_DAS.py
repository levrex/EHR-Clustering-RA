
# Import relevant modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

INPUT_DATA = sys.argv[1]
EXPORT_DATA = sys.argv[2]
SCHEDULE = sys.argv[3]
METADATA = sys.argv[4]

print('INPUT_DATA=', INPUT_DATA, ';\nEXPORT_DATA=', EXPORT_DATA, ';\nSCHEDULE=', SCHEDULE, ';\nMETADATA=', METADATA )


# Define function to pivot data
def pivot_data(dataframe:pd.DataFrame, which: str, prefix:str="Pijn", JC_name:str="TJC" ) -> pd.DataFrame:
    df = dataframe[["PATNR","DATUM","STELLING","XANTWOORD","M_DATUM"]][dataframe["STELLING"] == which].copy()
    df['value'] = 1
    df = df.pivot_table(index=("PATNR","DATUM"), columns="XANTWOORD", values="value",fill_value=0)
    for col in list(set(l_das44)):
        if col not in df.columns:
            # add empty column (value=0) to beginning
            df.insert(0, col, 0)
    df = df.rename(columns=name_map)
    df["total "+which] = df.sum(axis=1)
    df[JC_name+"_28"] = df[l_das28].sum(axis=1)
    df[JC_name+"_44"] = df[l_das44].sum(axis=1)
    renaming = {x:prefix+"_"+"_".join(x.split(" ")) for x in df.columns[:-3]}
    df = df.rename(columns=renaming)
    return df.reset_index()

# Define functions to calculate disease activity
def calculate_DAS28(tjc, sjc, esr):
    """
    Calculate DAS28 with 3 variables : TJC, SJC and ESR (BSE)
    """
    #print(esr)
    if esr != 0:
        das28 = (0.56 * np.sqrt(tjc) + 0.28 * np.sqrt(sjc) + 0.70 * np.log(esr)) * 1.08 + 0.16
    else :
        print(esr, tjc, sjc)
        print(eql)
    #print(das28)
    return das28

def calculate_DAS44(tjc, sjc, esr):
    """
    Calculate DAS44 with 3 variables : RAI, SJC and ESR (BSE)
    """

    das44= (0.53938 * np.sqrt(tjc) + 0.0650 * (sjc) + 0.330 * np.log(esr)) + 0.224 
    return das44

# Define joints for DAS28 & DAS44
l_das28 = [
    "pols links", "pols rechts", "pip 2 linkerhand", "pip 2 rechterhand", "pip 3 linkerhand", "pip 3 rechterhand",
    "pip 4 linkerhand", "pip 4 rechterhand",  "pip 5 linkerhand", "pip 5 rechterhand",
    "mcp 1 links", "mcp 1 rechts", "mcp 2 links", "mcp 2 rechts", "mcp 3 links", "mcp 3 rechts",
    "mcp 4 links", "mcp 4 rechts", "mcp 5 links", "mcp 5 rechts", "ip links", "ip rechts",
    "schouder links", "schouder rechts", 'elleboog links', 'elleboog rechts', 'knie links', 'knie rechts'
]
l_das44 = l_das28 + [
    'sternoclaviculair links', 'sternoclaviculair rechts', 'acromioclaviaculair rechts', 'acromioclaviculair links',
    "pip 2 linkervoet", "pip 2 rechtervoet", "pip 3 linkervoet", "pip 3 rechtervoet",
    "pip 4 linkervoet", "pip 4 rechtervoet",  "pip 5 linkervoet", "pip 5 rechtervoet",
    "bovenste spronggewricht links", "onderste spronggewricht links",
    "bovenste spronggewricht rechts","onderste spronggewricht rechts"
]    

l_das44_t = ["Gezwollen_"+ "_".join(x.split()) for x in l_das44.copy()]

name_map = {'IP links': 'ip links', 
 'IP rechts':'ip rechts',
 'IP voet links':'ip linkervoet',
 'IP voet rechts':'ip rechtervoet',
 'Manubrio sternaal gewricht':'manubrio sternaal gewricht',
 'acromioclaviaculair L':'acromioclaviculair links',
 'acromioclaviaculair R':'acromioclaviaculair rechts',
 'bovenste spronggewicht links':'bovenste spronggewricht links',
 'cmc 1 links':'cmc 1 links',
 'cmc 1 rechts':'cmc 1 rechts',
 'dip 2 links':'dip 2 linkerhand',
 'dip 2 links voet':'dip 2 linkervoet',
 'dip 2 rechts':'dip 2 rechterhand',
 'dip 2 rechts voet':'dip 2 rechtervoet',
 'dip 3 links':'dip 3 linkerhand',
 'dip 3 links voet':'dip 3 linkervoet',
 'dip 3 rechts':'dip 3 rechterhand',
 'dip 3 rechts voet':'dip 3 rechtervoet',
 'dip 4 links':'dip 4 linkerhand',
 'dip 4 links voet':'dip 4 linkervoet',
 'dip 4 rechts':'dip 4 rechterhand',
 'dip 4 rechts voet':'dip 4 rechtervoet',
 'dip 5 links':'dip 5 linkerhand',
 'dip 5 links voet':'dip 5 linkervoet',
 'dip 5 rechts':'dip 5 rechterhand',
 'dip 5 rechts voet':'dip 5 rechtervoet',
 'Elleboog L':'elleboog links',
 'elleboog R':'elleboog rechts',
 'pip 2 links hand':'pip 2 linkerhand',
 'pip 2 links voet':'pip 2 linkervoet',
 'pip 2 rechts hand':'pip 2 rechterhand',
 'pip 2 rechts voet':'pip 2 rechtervoet',
 'pip 3 links hand':'pip 3 linkerhand',
 'pip 3 links voet':'pip 3 linkervoet',
 'pip 3 rechts hand':'pip 3 rechterhand',
 'pip 3 rechts voet':'pip 3 rechtervoet',
 'pip 4 links hand':'pip 4 linkerhand',
 'pip 4 links voet': 'pip 4 linkervoet',
 'pip 4 rechts hand':'pip 4 rechterhand',
 'pip 4 rechts voet':'pip 4 rechtervoet',
 'pip 5 links hand':'pip 5 linkerhand',
 'pip 5 links voet':'pip 5 linkervoet',
 'pip 5 rechts hand':'pip 5 rechterhand',
 'pip 5 rechts voet':'pip 5 rechtervoet',
 'pols L':'pols links',
 'pols R':'pols rechts',
 'schouder L':'schouder links',
 'schouder R':'schouder rechts',
 'sternoclaviculair L':'sternoclaviculair links',
 'sternoclaviculair R':'sternoclaviculair rechts',
 'tarsometatarsaal L':'tarsometatarsaal Links',
 'tarsometatarsaal R':'tarsometatarsaal Rechts',
 'temporomandibulair L':'temporomandibulair links',
 'temporomandibulair R':'temporomandibulair rechts'}



# Import Mannequin data with Sedimentation rate (BSE)
################################################### UPDATE ####################################################
data = pd.read_csv(r'%s' % INPUT_DATA, sep='|', parse_dates=True,)
data = data.sort_values(by=['PATNR', 'DATUM_A'])

# Remove weird data artefacts
data = data[~((data['STELLING']=='Pijn') & (data['XANTWOORD'].str.contains('OBJECTID')==True))]
data = data[~((data['STELLING']=='Zwelling') & (data['XANTWOORD'].str.contains('OBJECTID')==True))]

# Use 'DATUM_A' instead of 'DATUM' as it is the actual date
data["M_DATUM"] = data.DATUM.copy()
data.DATUM = pd.to_datetime(data.DATUM_A, format='%Y-%m-%d')
data['DATUM'] = data['DATUM'].dt.date

# Link to schedule appointments
l_cols = ['subject_Patient_value', 'description', 'created_date']

consults = pd.read_csv(r'%s' % SCHEDULE, sep=';', parse_dates=True, header=None)
consults.columns = ['type1_code', 'description', 'subject_Patient_value', 'period_start', 'period_start_date', 'period_start_time', 'period_end', 'period_end_date', 'period_end_time']
consults = consults.rename(columns={'period_start_date' : 'created_date', 'period_start_time' : 'created_Time'})
consults['created_date']= pd.to_datetime(consults['created_date'], format='%Y-%m-%d')

consults = consults.sort_values(by=['created_date'])
consults = consults.rename(columns={'subject_Patient_value' : 'PATNR', 'created_date' : 'DATUM'})
consults['DATUM']= pd.to_datetime(consults['DATUM'], format='%Y-%m-%d')
data['DATUM']= pd.to_datetime(data['DATUM'], format='%Y-%m-%d')

data = consults.merge(data, how="inner", on=["PATNR","DATUM"])

### Transform mannequin info from long to wide
# - For both swollen and painful joints
ggdf = pivot_data(dataframe=data, which="Zwelling", prefix= "Gezwollen", JC_name="SJC")
pgdf = pivot_data(dataframe=data, which="Pijn", prefix= "Pijn", JC_name="TJC")

# Combine swollen & pain into one supertable
dataset = pgdf.merge(ggdf, how="outer", on=["PATNR","DATUM"]).fillna(0)

# Export mannequin info
#dataset.to_csv("mannequine_file.csv", index=False)

# Add exception whereby patient doesnt have any tender or swollen joints
df = data[["PATNR","DATUM","STELLING","XANTWOORD","M_DATUM"]][data["STELLING"] == "Geen pijnlijke gewrichten"].copy()
df["value"] = 1
test_pg = df.pivot_table(index=("PATNR","DATUM"), columns="STELLING", values="value",fill_value=0)
df = data[["PATNR","DATUM","STELLING","XANTWOORD","M_DATUM"]][data["STELLING"] == "Geen gezwollen gewrichten"].copy()
df["value"] = 1
test_gg =df.pivot_table(index=("PATNR","DATUM"), columns="STELLING", values="value",fill_value=0)
test_merge = test_gg.merge(test_pg, how="outer",left_index=True, right_index=True).fillna(0)

dataset = dataset.merge(test_merge, how="outer", on=["PATNR","DATUM"]).fillna(0).drop(["Geen gezwollen gewrichten", "Geen pijnlijke gewrichten"],axis=1)

# Acquire DAS 44 
d44 = data[data["STELLING"] == "DAS 44"][["PATNR","DATUM","STELLING","ANTWOORD"]]
d44["value"] = d44.ANTWOORD.str.rstrip('"').str.replace(',', '.').str.rsplit("VALUE1=",expand=True)[1].astype(np.float64)

# Acquire ESR
BSE = data[data["STELLING"] == "BSE"][["PATNR","DATUM","STELLING","ANTWOORD"]]
BSE["bse"] = BSE.ANTWOORD.astype(np.float64)
BSE = BSE.drop("STELLING", axis=1).drop("ANTWOORD", axis=1)

# Combine ESR with Mannequin information
TOL_BEFORE = 14
TOL_AFTER = 2

df_back = pd.merge_asof(dataset[['PATNR', 'DATUM']].sort_values(by='DATUM'), BSE.sort_values(by='DATUM'), by='PATNR', on='DATUM', tolerance=pd.Timedelta(days=TOL_BEFORE), direction='backward') #
df_forward = pd.merge_asof(dataset[['PATNR', 'DATUM']].sort_values(by='DATUM'), BSE.sort_values(by='DATUM'), by='PATNR', on='DATUM', tolerance=pd.Timedelta(days=TOL_AFTER), direction='forward')

dataset['bse'] = df_forward['bse'].fillna(df_back['bse'])

# Calculate nr of elapsed days
df_treat = pd.read_csv('%s' % METADATA, sep='|')
new_pat = dict(zip(df_treat.patnr, df_treat.FirstConsult)) # First consult

# Determine first consult data
dataset['PEC']= dataset['PATNR'].apply(lambda x: new_pat[x] if x in new_pat.keys() else np.nan)
dataset['PEC'] = pd.to_datetime(dataset['PEC'], format='%Y-%m-%d')
dataset = dataset[~dataset['PEC'].isna()].copy()
dataset['days'] = ((dataset['DATUM'] - dataset['PEC']) / np.timedelta64(1, 'D')).astype(int)

# Create relevant selection
l_pass = ['SJC_28', 'SJC_44','TJC_28', 'TJC_44', 'DATUM', 'PATNR', 'total Zwelling', 'total Pijn', 'bse', 'days', 'PEC']
das_df = dataset[l_pass].copy()
das_df.rename(columns= {'PATNR' : 'patnr','DATUM' : 'date'})


# Calculate DAS
das_df['DAS28(3)'] = das_df.apply(lambda x : calculate_DAS28(x['TJC_28'],x['SJC_28'] ,x['bse'] ), axis=1)
das_df['DAS44'] = das_df.apply(lambda x : calculate_DAS44(x['TJC_44'],x['SJC_44'] ,x['bse'] ), axis=1)

# Export DAS 
das_df = das_df.rename(columns={'SJC_28': 'SJC', 'TJC_28': 'TJC', 'bse' : 'ESR', 'PATNR':'patnr', 'DATUM': 'date', 'DAS28' : 'DAS28(3)'})
das_df = das_df.dropna()
das_df.to_csv(EXPORT_DATA, sep = ';', index=False)