import re
import math
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numpy.linalg import norm
#import umap
import unicodedata
import pattern.nl as patNL
import pattern.de as patDE
import pattern.en as patEN



def getStartTreatmentDate(df_med, d_pseudo_pat, d_pseudo_date, no_drug_window=6):
    """  
    Description:
    This function extracts the date where a patient takes medication for the first time
    (after the first consult)
    
    Input:
        df_med = dataframe with medicator information
        d_pseudo_pat = dictionary whereby pseudo ids are linked to patient ids 
        d_pseudo_date = dictionary whereby pseudo ids are linked to the first consult date
        no_drug_window = ensure there is no drug prior to the picked baseline (default = 6 months)
            - Otherwise, it is no true baseline because the our measures are then biased by
                drug interference
    Output:
        df_treat = dataframe with the date of the first treatment after the specified date of baseline
    """
    c_close = 0
    def nearestTreatment_LookBehind(items, pivot):
        
        items = [item for item in items if item < pd.to_datetime(pivot, format='%Y-%m-%d', errors='ignore') ]
        #print(items)
        if items == []:
            return np.nan
        else : 
            return min(items, key=lambda x: abs(x - pivot))
    
    df_treat = pd.DataFrame(columns=['pseudoId', 'patnr', 'index', 'Drug', 'Instruction', 'FirstConsult', 'Lookahead_Treatment', 'Lookbehind_Treatment', 'Lookahead_Prednison', 'Lookahead_DMARD', 'FILTER_RX_NA', 'FILTER_RX_NA_BASELINE']) 
    
    
    for pid in d_pseudo_pat.keys():
        no_med = True # register if medication is found
        missing_baseline = False
        pat = d_pseudo_pat[pid]
        
        ix = 0
        sub_df = df_med[df_med['PATNR']==pat].copy()
        sub_df = sub_df.sort_values(by='periodOfUse_valuePeriod_start_date')
        
        prednison_df = sub_df[sub_df['ATC_display'].isin(['PREDNISOLONE', 'METHYLPREDNISOLONE'])].copy()
        no_prednison_df = sub_df[~sub_df['ATC_display'].isin(['PREDNISOLONE', 'METHYLPREDNISOLONE'])].copy()
        for date in list(sub_df['periodOfUse_valuePeriod_start_date']):
            
            if pd.to_datetime(date, format='%Y-%m-%d', errors='ignore') >=  pd.to_datetime(d_pseudo_date[pid], format='%Y-%m-%d', errors='ignore') :
                sub_df['periodOfUse_valuePeriod_end_date'] = pd.to_datetime(sub_df['periodOfUse_valuePeriod_end_date'], format='%Y-%m-%d', errors='ignore')
                sub_df['periodOfUse_valuePeriod_start_date'] = pd.to_datetime(sub_df['periodOfUse_valuePeriod_start_date'], format='%Y-%m-%d', errors='ignore')
                lookbehind_end = nearestTreatment_LookBehind(sub_df['periodOfUse_valuePeriod_end_date'],pd.to_datetime(d_pseudo_date[pid], format='%Y-%m-%d', errors='ignore'))
                lookbehind_start = nearestTreatment_LookBehind(sub_df['periodOfUse_valuePeriod_start_date'],pd.to_datetime(d_pseudo_date[pid], format='%Y-%m-%d', errors='ignore'))
                if type(lookbehind_end) == float and type(lookbehind_start) == float:
                    lookbehind_date = np.nan
                elif type(lookbehind_end) == float:
                    lookbehind_date =lookbehind_start
                elif type(lookbehind_start) == float:
                    lookbehind_date =lookbehind_end
                else :
                    lookbehind_date = max(lookbehind_end,lookbehind_start)
                if type(lookbehind_date) != float: # nan
                    if lookbehind_date > pd.to_datetime(d_pseudo_date[pid], format='%Y-%m-%d', errors='ignore') - pd.DateOffset(months=no_drug_window): 
                        missing_baseline = True # ensure medication hasn't started (looking 6 months prior to consult)
                
                # Acquire first date of prednison
                prednison_df = prednison_df[pd.to_datetime(prednison_df['periodOfUse_valuePeriod_start_date'], format='%Y-%m-%d', errors='ignore') >= pd.to_datetime(date, format='%Y-%m-%d', errors='ignore')].sort_values(by='periodOfUse_valuePeriod_start_date')
                if len(prednison_df) > 0 :
                    first_pred = prednison_df['periodOfUse_valuePeriod_start_date'].iloc[0]
                else :
                    first_pred = np.nan
                    
                # Acquire first date of DMARD
                no_prednison_df = no_prednison_df[pd.to_datetime(no_prednison_df['periodOfUse_valuePeriod_start_date'], format='%Y-%m-%d', errors='ignore') >= pd.to_datetime(date, format='%Y-%m-%d', errors='ignore')].sort_values(by='periodOfUse_valuePeriod_start_date')
                if len(no_prednison_df) > 0 :
                    first_dmard = no_prednison_df['periodOfUse_valuePeriod_start_date'].iloc[0]
                else :
                    first_dmard = np.nan
                    
                df_treat.loc[len(df_treat)] = [pid, sub_df['PATNR'].iloc[ix], sub_df.iloc[ix].name, sub_df['ATC_display'].iloc[ix], sub_df['dosageInstruction_text'].iloc[ix], pd.to_datetime(d_pseudo_date[pid], format='%Y-%m-%d', errors='ignore'), date, lookbehind_date, first_pred, first_dmard, False, missing_baseline]
                no_med = False
                break 
                
                
            ix += 1
            
        if len(sub_df) == 0: # These patients have no medication information
            df_treat.loc[len(df_treat)] = [pid, pat, -1, np.nan, np.nan, pd.to_datetime(d_pseudo_date[pid], format='%Y-%m-%d', errors='ignore'), np.nan, np.nan, np.nan, np.nan, True, True]
        elif no_med: # These patients have medication information at some point, but not in close proximity to the baseline
            df_treat.loc[len(df_treat)] = [pid, pat, -1, np.nan, np.nan, pd.to_datetime(d_pseudo_date[pid], format='%Y-%m-%d', errors='ignore'), np.nan, np.nan, np.nan, np.nan, False, True]
    
    return df_treat

def getStartTreatmentDateMETEOR(df_med, d_pat_date, no_drug_window=6):
    """  
    Description:
    This function extracts the date where a patient takes medication for the first time
    (after the first consult)
    
    This function is designed for METEOR data, whereby the first consult is not ambigious.
    
    Input:
        df_med = dataframe with medicator information
        d_pat_date = dictionary whereby patient ids are linked to the first consult date
        no_drug_window = ensure there is no drug prior to the picked baseline (default = 6 months)
            - Otherwise, it is no true baseline because the our measures are then biased by
                drug interference
    Output:
        df_treat = dataframe with the date of the first treatment after the specified date of baseline
    """
    c_close = 0
    def nearestTreatment_LookBehind(items, pivot):
        
        items = [item for item in items if item < pd.to_datetime(pivot, format='%Y-%m-%d', errors='ignore') ]
        #print(items)
        if items == []:
            return np.nan
        else : 
            return min(items, key=lambda x: abs(x - pivot))
    
    df_treat = pd.DataFrame(columns=['patnr', 'index', 'Drug', 'Instruction', 'FirstConsult', 'Lookahead_Treatment', 'Lookbehind_Treatment', 'Lookahead_Prednison', 'FILTER_RX_NA', 'FILTER_RX_NA_BASELINE']) 
    
    
    for pat in df_med['PATNR'].unique():
        no_med = True # register if medication is found
        missing_baseline = False
        
        ix = 0
        sub_df = df_med[df_med['PATNR']==pat].copy()
        sub_df = sub_df.sort_values(by='periodOfUse_valuePeriod_start_date')
        
        prednison_df = sub_df[sub_df['ATC_display'].isin(['PREDNISOLONE', 'METHYLPREDNISOLONE'])].copy()
        for date in list(sub_df['periodOfUse_valuePeriod_start_date']):
            
            if pd.to_datetime(date, format='%Y-%m-%d', errors='ignore') >=  pd.to_datetime(d_pat_date[pat], format='%Y-%m-%d', errors='ignore') :
                sub_df['periodOfUse_valuePeriod_end_date'] = pd.to_datetime(sub_df['periodOfUse_valuePeriod_end_date'], format='%Y-%m-%d', errors='ignore')
                sub_df['periodOfUse_valuePeriod_start_date'] = pd.to_datetime(sub_df['periodOfUse_valuePeriod_start_date'], format='%Y-%m-%d', errors='ignore')
                lookbehind_end = nearestTreatment_LookBehind(sub_df['periodOfUse_valuePeriod_end_date'],pd.to_datetime(d_pat_date[pat], format='%Y-%m-%d', errors='ignore'))
                lookbehind_start = nearestTreatment_LookBehind(sub_df['periodOfUse_valuePeriod_start_date'],pd.to_datetime(d_pat_date[pat], format='%Y-%m-%d', errors='ignore'))
                if type(lookbehind_end) == float and type(lookbehind_start) == float:
                    lookbehind_date = np.nan
                elif type(lookbehind_end) == float:
                    lookbehind_date =lookbehind_start
                elif type(lookbehind_start) == float:
                    lookbehind_date =lookbehind_end
                else :
                    lookbehind_date = max(lookbehind_end,lookbehind_start)
                if type(lookbehind_date) != float: # nan
                    if lookbehind_date > pd.to_datetime(d_pat_date[pat], format='%Y-%m-%d', errors='ignore') - pd.DateOffset(months=no_drug_window): 
                        missing_baseline = True # ensure medication hasn't started (looking 6 months prior to consult)
                
                # Acquire first date of prednison
                prednison_df = prednison_df[pd.to_datetime(prednison_df['periodOfUse_valuePeriod_start_date'], format='%Y-%m-%d', errors='ignore') >= pd.to_datetime(date, format='%Y-%m-%d', errors='ignore')].sort_values(by='periodOfUse_valuePeriod_start_date')
                if len(prednison_df) > 0 :
                    first_pred = prednison_df['periodOfUse_valuePeriod_start_date'].iloc[0]
                else :
                    first_pred = np.nan
                df_treat.loc[len(df_treat)] = [sub_df['PATNR'].iloc[ix], sub_df.iloc[ix].name, sub_df['ATC_display'].iloc[ix], sub_df['dosageInstruction_text'].iloc[ix], pd.to_datetime(d_pat_date[pat], format='%d-%m-%Y', errors='ignore'), date, lookbehind_date, first_pred, False, missing_baseline]
                no_med = False
                break 
            ix += 1
            
        if len(sub_df) == 0: # These patients have no medication information
            df_treat.loc[len(df_treat)] = [ pat, -1, np.nan, np.nan, pd.to_datetime(d_pat_date[pat], format='%Y-%m-%d', errors='ignore'), np.nan, np.nan, np.nan, True, True]
        elif no_med: # These patients have medication information at some point, but not in close proximity to the baseline
            df_treat.loc[len(df_treat)] = [ pat, -1, np.nan, np.nan, pd.to_datetime(d_pat_date[pat], format='%Y-%m-%d', errors='ignore'), np.nan, np.nan, np.nan, False, True]
    
    return df_treat

def removeAccent(text):
    """
    This function removes the accent of characters from the text.

    Variables:
        text = text to be processed
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return text

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


def lemmatizingText(sentence, lan='en'):
    """
    This function normalizes words with the pattern.nl package. 
    Lemmatisation returns words to the base form. The base form
    should be a valid word in the language.

    Example: Walking, Walks and Walked are all translated to 
        Walk
        
    Input: 
        sentence = written text from an EHR record or another
            Natural Language type record (str)
    """
    if lan == 'nl':
        return ' '.join(patNL.Sentence(patNL.parse(sentence, lemmata=True)).lemmata)
    elif lan == 'en':
        return ' '.join(patNL.Sentence(patNL.parse(sentence, lemmata=True)).lemmata)


def simpleCleaning(sentence, lemma=False): # Keep in mind: this function removes numbers
    """
    Remove special characters that are not relevant to 
    the interpretation of the text
    
    Input:
        sentence = free written text from EHR record
        lemma = lemmatize the text
    Output :
        processed sentence (lemmatized depending on preference)
    """
    sticky_chars =r'([!#,.:";@\-\+\\/&=$\]\[<>\'^\*`â€™\(\)])' #r'([!#,.:";@\-\+\\/&=$\]\[<>\'^\*`â€™\(\)])'
    sentence = re.sub(sticky_chars, r' ', sentence)
    sentence = sentence.lower()
    return sentence
    
    
def reformat_ddrA(df, pat_col='PATNR', time_col='DATUM', aggregate=False): # patient_data, 
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
        else :
            o_df.at[o_row, feature] = value
            #print(eql)
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
                result = df_sub['uitslag_text'].iloc[0]
                if result not in ['-volgt-', '@volgt', np.nan] : # stopgezet?
                    d_val[pat] = result
        #else:
        #    d_val[pat] = np.nan
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
                val_str = 'Dubieus'
        elif val_str in ['nan', 'Bepaling niet ui']: 
            val_str = np.nan
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
                val_str = 'Dubieus'
        elif val_str in ['nan', 'Bepaling niet ui']: 
            val_str = np.nan
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
        
    #print(f"It's pd.isna  : {pd.isna(unit)}")
    #print(f"It's np.isnan  : {np.isnan(unit)}")
    #print(f"It's math.isnan : {math.isnan(unit)}")
    
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
        if '10^6/L' in unit:
            value = float(value) * 10**-3 # convert to 10^9
            unit = '10^9/L'
        elif 'ug/L' in unit or 'µg/L' in unit:
            value = float(value) * 10**-6 # convert to g/L
            unit = 'g/L'
        elif unit == 'Ratio':
            unit = np.nan
        
        #if value in ['Dubieus', 'Zwak pos.', 'Zwak Positief', 'Negatief']:
        #    value = 0
        #elif value in ['Positief', 'Sterk pos.', 'Sterk Positief']:
        #    value = 1
        #elif value in ['-volgt-', 'Stopgezet', '@volgt']:
        #    value = np.nan
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
    """
    print("\nProcessing lab data...") # PseudoID # patient_id'
    # Sort df - Keep the first measure -> cause we want to get the baseline characteristics
    df.sort_values([pat_col, time_col], ascending=False, ignore_index=True, inplace=True)
    # New empty output df
    o_df = pd.DataFrame(columns=['pseudoId', 'time'])
    o_row = -1
    pseudo_id = 0
    o_time = 0

    # Loop through df
    for row in range(len(df.index)):
        # Print progress information
        if row % round(len(df.index) / 8) == 0:
            print(f"Progress: row {row} / {len(df.index)} ({round(row / len(df.index) * 100, 1)}%)")

        # Add row if (a) a new pseudo_id is registered, or (b) a new timepoint is registered
        if pseudo_id != df[pat_col].iloc[row]  : # or o_time != df[time_col].iloc[row]
            o_row += 1
            pseudo_id = df[pat_col].iloc[row]
            o_time = df[time_col].iloc[row]

        # Feature names in the long format are built from a description and unit column
        # Sometimes the unit is already in the description, in that case ignore the unit column
        descr = df['test_naam_omschrijving'].iloc[row]
        value = df['uitslag_value'].iloc[row] # test_naam_omschrijving
        unit = df['uitslag_unit'].iloc[row]
        if str(descr).endswith(str(unit)) or str(descr).endswith(f'({unit})'):
            unit = np.nan
        
        feature, value = fuzzy_feature(descr, unit, value)
        
        # Make absolute time point relative to first symptoms day
        #time = abs_to_rel_date(patient_data['symptoms_date'].loc[pat_col], o_time)
        time = o_time

        # Add new column to output dataframe if feature has not yet been made
        if feature not in o_df.columns:
            o_df[feature] = np.nan

        # Populate row in output dataframe
        o_df.at[o_row, 'pseudoId'] = pseudo_id
        o_df.at[o_row, feature] = value
        o_df.at[o_row, 'time'] = time

    return o_df
