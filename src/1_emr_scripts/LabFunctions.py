import re
import math
import numpy as np
import pandas as pd
    
def processArtefactsXML(entry):
    """
    Removes XML artefacts with a mapping function
    
    Input : 
        entry - Free written text entry from Electronic Health
            record (EHR)
    Output:
        entry - processed text field
    """
    correction_map ={'ã«' : 'e', 'ã¨' : 'e', 'ã¶': 'o', '\r' : ' ', '\n' : ' ', '\t': ' ', '·' : ' ', 
                     'ã©' : 'e', 'ã¯' : 'i', 'ãº':'u', 'ã³' : 'o', '\xa0' : ' '}
    for char in correction_map.keys():
        entry = entry.replace(char, correction_map[char])
    return entry
    
def reformat_ddrA(df, pat_col='PATNR', time_col='DATUM', aggregate=False): # patient_data, # DATUM_A
    """
    Cast DDR_A questionaire file from long to wide format. 
    Entries are merged on patient number.
    
    Input:
        df = dataframe (default HIX-output)
        pat_col = the patient identifier column
        time_col = the column with the date
        aggregate = whether or not to aggregate the text
    Output:
        o_df = output dataframe where each variable is its own column
    """
    print("\nProcessing lab data...") # PseudoID # patient_id'
    # Sort df - Keep the first measure -> cause we want to get the baseline characteristics
    df.sort_values([pat_col, time_col], ascending=True, ignore_index=True, inplace=True)
    # New empty output df
    o_df = pd.DataFrame(columns=['patnr', 'time'])
    o_row = -1
    pseudo_id = 0
    o_time = 0

    # Loop through df
    for row in range(len(df.index)):
        # Print progress information
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")

        # Add row if (a) a new pseudo_id is registered
        if pseudo_id != df[pat_col].iloc[row]  : # or o_time != df[time_col].iloc[row]
            o_row += 1
            pseudo_id = df[pat_col].iloc[row]
            o_time = df[time_col].iloc[row]

        # Feature names in the long format are built from a description and unit column
        # Sometimes the unit is already in the description, in that case ignore the unit column
        descr = df['STELLING'].iloc[row]
        value = df['XANTWOORD'].iloc[row] 
        
        feature = f"{descr}"
        
        # Make absolute time point relative to first symptoms day
        #time = abs_to_rel_date(patient_data['symptoms_date'].loc[pat_col], o_time)
        time = o_time

        # Add new column to output dataframe if feature has not yet been made
        if feature not in o_df.columns:
            o_df[feature] = ''

        # Populate row in output dataframe
        o_df.at[o_row, 'patnr'] = pseudo_id
        if type(o_df.at[o_row, feature]) != float and o_df.at[o_row, feature] != '' and aggregate==True:
            o_df.at[o_row, feature] += ' [END_RECORD] ' + value
        elif o_df.at[o_row, feature] == '' or type(o_df.at[o_row, feature]) == float:
            o_df.at[o_row, feature] = value
        o_df.at[o_row, 'time'] = time
        
    return o_df

def create_dict_pat(df, val=['Anti-CCP']):
    """
    This function is called to retrieve the boolean values, since
    the boolean values are stored in a different column.
    
    Input:
        df = dataframe (default HIX-output)
        val = boolean value (binary label assigned based on the quantity)
    Output:
        d_val = dictionary with the lab quantity
    """
    d_val = {}
    for pat in df['pseudoId'].unique():
        df_sub = df[((df['pseudoId']==pat) & (df['test_naam_omschrijving'].isin(val)))]
        if len(df_sub) > 0:
            for i in range(len(df_sub)):
                result = df_sub['uitslag_text'].iloc[i]
                if result not in ['-volgt-', '@volgt', np.nan, 'gestopt'] : # stopgezet?
                    d_val[pat] = result
    return d_val

def use_max_lab(row, col_test=['BSE'], col_val='uitslag_value', col_alt='uitslag_text', col_desc='test_naam_omschrijving'):
    """
    If Sedimentation is above the maximal value, use the maximal cut-off value

    MAX for BSE: > 140.0
    
    Input:
        row = row from lab dataframe
        col_test = lab value to impute
        col_val = column where the result as a value is found
        col_alt = specify column where the max value is mentioned
        col_desc = column where the type of test is described
    
    Output:
        val = Either initial value or value inferred based on max cut-off
    """
    val = row[col_val]
    if row[col_desc] in col_test:
        try :
            val_str = str(row[col_alt])
            if re.sub("[^\d\.]", "", val_str) != '': # if numbers found
                val_str = re.sub("[^\d\.]", "", val_str)
                return val_str
            else :
                return val
        except:
            return val
    else :
        return val
    

def infer_RF(val):
    """
    If RF is expressed as a value -> infer the status (True/False) 
    by using the reference range.
    
    Negative: < 3.5
    Dubious: 3.5 - 5.0
    Positive: > 5.0
    
    Output:
        val = RF status (either Positive, Negative or Ambiguous)
    """
    try :
        val_str = str(val)
        if re.sub("[^\d\.]", "", val_str) != '': # if numbers found
            val_str = re.sub("[^\d\.]", "", val_str)
            if float(val_str) >= 5:
                val_str = 'Positief'
            elif float(val_str) < 3.5 : # or under 3.5
                val_str= 'Negatief'
            else  : # or under 3.5
                val_str = 'Negatief' #'Dubieus'
        elif val_str in ['nan', 'Bepaling niet ui', 'Aanvraag reeds u', 'Geen materiaal o']: 
            val_str = np.nan
        elif val_str in 'Dubieus':
            val_str = 'Negatief'
        return val_str
    except:
        return val

def infer_aCCP(val):
    """
    If CCP is expressed as a value -> infer the status (True/False) 
    by using the reference range.
    
    Negative: < 7
    Dubious: 7 - 9
    Positive: > 9
    
    Output:
        val = CCP status (either Positive, Negative or Ambiguous)
    """
    try :
        val_str = str(val)
        if re.sub("[^\d\.]", "", val_str) != '': # if numbers found
            val_str = re.sub("[^\d\.]", "", val_str)
            if float(val_str) >= 9:
                val_str = 'Positief'
            elif float(val_str) < 7 : # or under 7
                val_str = 'Negatief'
            else  : # or under 7
                val_str = 'Negatief' #'Dubieus'
        elif val_str in ['nan', 'Bepaling niet ui', 'Aanvraag reeds u', 'Geen materiaal o']: 
            val_str = np.nan
        elif val_str in 'Dubieus':
            val_str = 'Negatief'
        return val_str
    except:
        return val
    
def infer_SSA(val):
    """
    If SSA (alkaline phosphate) is expressed as a value -> 
    infer the status (True/False) by using the reference range.
    
    Negative: < 20
    Weak Pos: 20-39
    Positive: > 40
    Strong Pos: > 80

    Output:
        val = SS-A/Ro status (either Positive or Negative)
    """
    try :
        val_str = str(val)
        if re.sub("[^\d\.]", "", val_str) != '': # if numbers found
            val_str = re.sub("[^\d\.]", "", val_str)
            if float(val_str) < 20 :
                val_str = 'Negatief'
            elif float(val_str) < 40: 
                val_str = 'Zwak Positief'
            elif float(val_str) < 80: 
                val_str= 'Positief'
            else :
                val_str = 'Sterk Positief'
        return val_str
    except:
        return val
    
def infer_AntiENA(val):
    """
    If anti-ENA is expressed as a value -> infer the status (True/False)
    by using the reference range.
    
    Negative: < 10
    Dubious: 10-15
    Positive: > 15

    Output:
        val = anti-ENA status (either Positive or Negative)
    """
    try :
        val_str = str(val)
        if re.sub("[^\d\.]", "", val_str) != '': # if numbers found
            val_str = re.sub("[^\d\.]", "", val_str)
            if float(val_str) < 10 :
                val_str = 'Negatief'
            elif float(val_str) < 15: 
                val_str = 'Dubieus'
            else: 
                val_str= 'Positief'
        return val_str
    except:
        return val
    
def LabelEncoder(val, include_na=True):
    """
    Standardize the categorical variables to numerical values
    
    where:
        0 = Negative
        1 = Positive
        2 = Missing
    
    Input:
        val = Categorical value 
        include_na = whether or to include missing values
            - in case the missingness contains information
            - you should keep these values!
    Output:
        val = codified categorical value
    """
    
    if val in ['Positief', 'Sterk pos.', 'Sterk pos', 'Sterk Positief']:
        val = 1
    elif val in ['Dubieus', 'Negatief', 'Zwak pos.', 'Zwak pos', 'Zwak Positief']:
        val= 0
    elif include_na:
        if val in ['Stopgezet', '-volgt-', '@volgt', 'Niet doorgeg', 'Niet het jui', 
                   'gestopt', 'Bepaling niet ui', ' ', 'Geen analyse mog', 'Te weinig materi', 
                   'Geen materiaal o', 'te weinig ma', 'gestopt',  'Geen analyse', 'geen uitslag', np.nan]:
            val = 2
    return val

def fuzzy_feature(descr, unit, value):
    """
    Standardize / Normalize the lab values
    
    Input:
        descr =  description of Lab test
        unit = unit used for lab test
        value = resulting value
    Output:
        feature = feature name (Format: descr (unit))
    """
    if type(descr) != float:
        descr = descr.strip()
        if descr in ['Ht', 'Hematocriet [CKCL]']:
            descr = 'Hematocriet'
        elif descr in ['Hb', 'Hemoglobine [CKCL]', 'Hemoglobine (art)']:
            descr = 'Hemoglobine'
        elif '#' in descr:
            descr = descr[:-2]
        elif 'Kwant.' in descr:
            descr = descr[:-6]
        elif 'ANF' in descr:
            descr = 'Anti nucleaire antistoffen (ANA)'
        elif 'C-Reaktief Proteïne' in descr:
            descr = 'C-Reactief Proteine'
        if ' ' == descr[-1]:
            descr = descr[:-1]
            
    if pd.isnull(unit) == False:
        unit = unit.replace("x", "")
        unit = unit.strip()
        if unit == 'Mm':
            unit = 'mm'
        elif unit == 'Mmol/L' or unit == 'mmol/L' or unit == 'mmol/l':
            unit = 'mmol/L'
        elif unit == 'umol/L':
            unit = 'µmol/L'
        elif unit == 'Fmol' :
            unit = 'fmol'
        elif unit == 'FL' or unit == 'fL':
            unit = 'fl'
        elif unit == '/ul' or unit== '/uL':
            unit = '/µl'
        if '10^6/L' in unit or '10*6/L' in unit:
            value = float(value) * 10**-3 # convert to 10^9
            unit = '10^9/L'
        elif 'ug/L' in unit or 'µg/L' in unit:
            value = float(value) * 10**-6 # convert to g/L
            unit = 'g/L'
        elif unit == 'Ratio':
            unit = np.nan

    feature = f"{descr}{f' ({unit})' if not pd.isnull(unit) else ''}"
    return feature, value
        

def reformat_lab(df, pat_col='pseudoId', time_col='Monster_Afname_Datumtijd'):
    """
    Cast raw LAB file from long to wide format. 
    Entries are merged on pseudoId (patnr + visit nr).
    
    Input:
        df = dataframe (default HIX-output)
        pat_col = the patient identifier column
        time_col = the column with the date
    Output:
        o_df = output dataframe where each lab value is its own column
    
    
    Todo: add step to ensure it falls within max or min
    """
    
    l_serology = ['Anti-CCP', 'Anti-CCP Kwant.', 'IgM reumafactor', 'RF.M Elisa']
    print("\nProcessing lab data...") # PseudoID # patient_id'
    # Sort df - Keep the first measure -> cause we want to get the baseline characteristics
    df.sort_values([pat_col, time_col], ascending=True, ignore_index=True, inplace=True)
    # New empty output df
    o_df = pd.DataFrame(columns=['pseudoId', 'time', 'lastValAdded'])
    o_row = -1
    pseudo_id = 0
    o_time = 0
    
    d_feat = {}

    # Loop through df
    for row in range(len(df.index)):
        # Print progress information
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")
        
        o_time = df[time_col].iloc[row]
        
        # Add row if (a) a new pseudo_id is registered, or (b) a new timepoint is registered
        if pseudo_id != df[pat_col].iloc[row]: # or o_time != df[time_col].iloc[row]
            if pseudo_id == '304822378_1':
                print(d_feat)
                
            o_row += 1
            pseudo_id = df[pat_col].iloc[row]
            d_feat = {} # reset
        

        # Feature names in the long format are built from a description and unit column
        # Sometimes the unit is already in the description, in that case ignore the unit column
        descr = df['test_naam_omschrijving'].iloc[row]
        value = df['uitslag_value'].iloc[row] # test_naam_omschrijving
        unit = df['uitslag_unit'].iloc[row]
        if str(descr).endswith(str(unit)) or str(descr).endswith(f'({unit})'):
            unit = np.nan
        
        feature, value = fuzzy_feature(descr, unit, value)

        # Add new column to output dataframe if feature has not yet been made (to add later on)
        if feature not in o_df.columns:
            o_df[feature] = np.nan
        
        # Make absolute time point relative to first symptoms day
        #time = abs_to_rel_date(patient_data['symptoms_date'].loc[pat_col], o_time)
        if feature not in d_feat.keys():   
            d_feat[feature] = o_time
            # Populate row in output dataframe
            o_df.at[o_row, 'pseudoId'] = pseudo_id
            o_df.at[o_row, feature] = value
            if feature not in l_serology:
                o_df.at[o_row, 'time'] = o_time
            o_df.at[o_row, 'lastValAdded'] = feature
    return o_df