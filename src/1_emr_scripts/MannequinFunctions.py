import pandas as pd
import numpy as np
import re

def calculate_DAS28(tjc, sjc, esr):
    """
    Calculate DAS28 with 3 variables : TJC, SJC and ESR (BSE)
    """
    if esr != 0:
        das28 = (0.56 * np.sqrt(tjc) + 0.28 * np.sqrt(sjc) + 0.70 * np.log(esr)) * 1.08 + 0.16
    else :
        print(esr, tjc, sjc)
        print(eql)
    return das28

def calculate_DAS44(tjc, sjc, esr):
    """
    Calculate DAS44 with 3 variables : RAI, SJC and ESR (BSE)
    """

    das44= (0.53938 * np.sqrt(tjc) + 0.0650 * (sjc) + 0.330 * np.log(esr)) + 0.224 #0.224
    return das44

def getDAS(df_das, tol=31*3, verbose=False):
    """  
    Retrieve disease activity at baseline from the generated DAS table
    
    Input:
        df_das = table with calculated DAS per day
        tol = Time window to look from with respect to baseline. 
            The default tolerance is 3 months.
        verbose = whether or not to print progress (useful when debugging)
            
    Output: 
        d_das28 = dictionary with pseudoIds matched to the DAS28
        d_das44 = dictionary with pseudoIds matched to the DAS44
        d_totalFollowUp = dictionary that keeps track of the followup. This follow up 
            is based on whether or not outcome variables can be found)
    """
    
    def find_nearest_ix(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    d_das28 = {}
    d_das44 = {}
    d_totalFollowUp = {}

    for pid in df_das['pseudoId'].unique():
        sub_df = df_das[df_das['pseudoId']==pid].copy()

        # find das closest to baseline
        idx, days = find_nearest_ix(sub_df['new_days'], 0) 
        d_totalFollowUp[pid] = (pd.to_datetime(sub_df['date'].max(), format='%Y-%m-%d', errors='ignore') - pd.to_datetime(sub_df['date'].min(), format='%Y-%m-%d', errors='ignore')).days
        if abs(days) < tol: # accept 3 months of tolerance
            das28 = sub_df['DAS28(3)'].iloc[idx]
            das44 = sub_df['DAS44'].iloc[idx]
            if verbose : print(pid, round(das28,2), round(das44,2), days)
            d_das44[pid] = das44
            d_das28[pid] = das28
        else :
            if verbose : print('No DAS within baseline (+/- 3 mo) for', pid)
    return d_das28, d_das44, d_totalFollowUp

def count_special_joints_2(row):
    """ 
    In this function we create 5 new features:
        1. Big joint count
        2. Small joint count
        3. Symmetrical joint count
        4. Swollen joint count
        5. Tender joint count
    
    Few remarks: 
        1-2: Big/ joint distinction is made in accordance with 
             the ACR/ EULAR 2010 criteria. 
        3:   Symmetry is asserted based on an exact match for big joint, 
             but fuzzy match for the smaller joints. For example, we can 
             infer symmetry if a patient is affected in MTP-1 on the left and
             MTP-2 on the right.
        4-5: We calculated TJC and SJC ourselves because there is 
             quite some missingness otherwise.
    
    Input:
        row = row from the Mannequin dataset (pandas Series)
    Output:
        big_joints = Big joint count
        small_joints = Small joint count
        c_sym = Symmetrical joint count
        c_sjc = Swollen joint count
        c_tjc = Tender joint count
    
    """
    # Con 2010 criteria only identified small / big joints for a select view. 
    # -> we could say -> everything bigger than wrist should be included?
    
    l_2010_big = ['Pijn_schouder links', 'Pijn_schouder rechts', 'Pijn_Elleboog links',
             'Pijn_elleboog rechts', 'Pijn_heup links', 'Pijn_heup rechts', 
             'Pijn_knie links', 'Pijn_knie rechts', 
             'Zwelling_schouder links', 'Zwelling_schouder rechts', 'Zwelling_Elleboog links', 
             'Zwelling_elleboog rechts', 'Zwelling_heup links', 'Zwelling_heup rechts', 
             'Zwelling_knie links', 'Zwelling_knie rechts',
             'Pijn_onderste spronggewricht links', 'Pijn_onderste spronggewricht rechts', 'Pijn_bovenste spronggewicht links', 'Pijn_bovenste spronggewricht rechts',
             'Zwelling_onderste spronggewricht links', 'Zwelling_onderste spronggewricht rechts', 'Zwelling_bovenste spronggewicht links', 'Zwelling_bovenste spronggewricht rechts',

            'Zwelling_cervical spine', 'Pijn_cervical spine', 'Pijn_sacro-ileacaal gewricht links', 'Pijn_sacro-ileacaal gewricht rechts',
                 ] # Sacro-ileacaal has no 'Zwelling' version
    l_2010_small = ['Pijn_pols links', 'Pijn_pols rechts', 'Pijn_pip 2 links hand', 'Pijn_pip 2 links voet', 'Pijn_pip 2 rechts hand',
 'Pijn_pip 2 rechts voet', 'Pijn_pip 3 links hand', 'Pijn_pip 3 links voet', 'Pijn_pip 3 rechts hand', 'Pijn_pip 3 rechts voet',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 links voet', 'Pijn_pip 4 rechts hand', 'Pijn_pip 4 rechts voet', 'Pijn_pip 5 links hand',
 'Pijn_pip 5 links voet', 'Pijn_pip 5 rechts hand', 'Pijn_pip 5 rechts voet', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 
 'Pijn_mtp 2 links', 'Pijn_mtp 2 rechts', 'Pijn_mtp 3 links', 'Pijn_mtp 3 rechts', 'Pijn_mtp 4 links', 'Pijn_mtp 4 rechts',
 'Pijn_mtp 5 links', 'Pijn_mtp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts',
                   'Zwelling_pols links', 'Zwelling_pols rechts', 'Zwelling_pip 2 links hand', 'Zwelling_pip 2 links voet', 'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 2 rechts voet', 'Zwelling_pip 3 links hand', 'Zwelling_pip 3 links voet', 'Zwelling_pip 3 rechts hand', 'Zwelling_pip 3 rechts voet',
 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 links voet', 'Zwelling_pip 4 rechts hand', 'Zwelling_pip 4 rechts voet', 'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 links voet', 'Zwelling_pip 5 rechts hand', 'Zwelling_pip 5 rechts voet', 
 'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
 'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
 'Zwelling_mtp 2 links', 'Zwelling_mtp 2 rechts', 'Zwelling_mtp 3 links', 'Zwelling_mtp 3 rechts', 'Zwelling_mtp 4 links', 'Zwelling_mtp 4 rechts',
 'Zwelling_mtp 5 links', 'Zwelling_mtp 5 rechts', 'Zwelling_IP links', 'Zwelling_IP rechts', 
                    
                    'Pijn_IP voet links', 'Pijn_IP voet rechts', 'Zwelling_IP voet links', 'Zwelling_IP voet rechts',
                    'Pijn_cmc 1 links', 'Pijn_cmc 1 rechts','Zwelling_cmc 1 links', 'Zwelling_cmc 1 rechts',
                    'Zwelling_acromioclaviaculair links', 'Zwelling_acromioclaviaculair rechts', 'Pijn_acromioclaviaculair links', 'Pijn_acromioclaviaculair rechts',
                'Zwelling_dip 2 links','Zwelling_dip 2 links voet','Zwelling_dip 2 rechts', 'Zwelling_dip 2 rechts voet',
                 'Zwelling_dip 3 links', 'Zwelling_dip 3 links voet', 'Zwelling_dip 3 rechts', 'Zwelling_dip 3 rechts voet',
                 'Zwelling_dip 4 links',  'Zwelling_dip 4 links voet', 'Zwelling_dip 4 rechts', 'Zwelling_dip 4 rechts voet',
                 'Zwelling_dip 5 links', 'Zwelling_dip 5 links voet', 'Zwelling_dip 5 rechts',  'Zwelling_dip 5 rechts voet',
                 'Pijn_dip 2 links','Pijn_dip 2 links voet', 'Pijn_dip 2 rechts', 'Pijn_dip 2 rechts voet', 
                 'Pijn_dip 3 links','Pijn_dip 3 links voet', 'Pijn_dip 3 rechts', 'Pijn_dip 3 rechts voet', 
                 'Pijn_dip 4 links', 'Pijn_dip 4 links voet', 'Pijn_dip 4 rechts', 'Pijn_dip 4 rechts voet', 
                 'Pijn_dip 5 links', 'Pijn_dip 5 links voet', 'Pijn_dip 5 rechts', 'Pijn_dip 5 rechts voet',
                    'Zwelling_sternoclaviculair links', 'Zwelling_sternoclaviculair rechts', 'Pijn_sternoclaviculair links', 'Pijn_sternoclaviculair rechts',
                    'Zwelling_Manubrio sternaal gewricht','Pijn_Manubrio sternaal gewricht', 
                    'Zwelling_tarsometatarsaal links', 'Zwelling_tarsometatarsaal rechts', 'Pijn_tarsometatarsaal links', 'Pijn_tarsometatarsaal rechts',
                    'Zwelling_temporomandibulair links', 'Zwelling_temporomandibulair rechts','Pijn_temporomandibulair links', 'Pijn_temporomandibulair rechts',
                    'Pijn_mtp 1 links', 'Pijn_mtp 1 rechts', 'Zwelling_mtp 1 links', 'Zwelling_mtp 1 rechts',
                   ]
    l_DAS28 = ['Pijn_pols links', 'Pijn_pols rechts', 'Pijn_pip 2 links hand', 'Pijn_pip 2 rechts hand', 'Pijn_pip 3 links hand', 'Pijn_pip 3 rechts hand',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 rechts hand', 'Pijn_pip 5 links hand', 'Pijn_pip 5 rechts hand', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts', 'Pijn_schouder links', 'Pijn_schouder rechts', 'Pijn_Elleboog links','Pijn_elleboog rechts',
 'Pijn_knie links', 'Pijn_knie rechts', 
           
 'Zwelling_pols links', 'Zwelling_pols rechts', 'Zwelling_pip 2 links hand',  'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 3 links hand',  'Zwelling_pip 3 rechts hand', 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 rechts hand',  'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 rechts hand',  'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
 'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
 'Zwelling_knie links', 'Zwelling_knie rechts', 'Zwelling_schouder links', 'Zwelling_schouder rechts', 'Zwelling_Elleboog links',
  'Zwelling_elleboog rechts', 'Zwelling_IP links', 'Zwelling_IP rechts'
                   ]
    l_ignore = ['FirstConsult', 'patnr', 'pseudoId', 'Big joints', 'Small joints', 'Symmetrical joints', 'SJC', 'TJC', 'Zwelling_beiderzijds', 'Pijn_beiderzijds', 'Zwelling_ja', 'Pijn_ja'  ]
    
    #print(row)
    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]
    
    #row.index= row.index.str.lower()
        
    # write out abbreviations
    row.index= row.index.str.replace(' R$', ' rechts', regex=True)
    row.index= row.index.str.replace(' L$', ' links', regex=True)
    
    d = row.to_dict()
    set_big, set_small = [], []
    
    
    c_sjc = 0 # keep track of nr of swollen joints
    c_tjc = 0 # keep track of nr of tender joints
    c_sym = 0 # keep track of nr of symmetrical joints
    
    for key in d.keys(): # This complicated function is needed to ensure that tender/swollen joints aren't counted twice
        val = d[key]
        if val == 1.0 and key in l_2010_big : # Disclaimer: only consider 'ACR 2010' joints
            if 'Zwelling' in key: 
                set_big.append(remove_prefix(key, 'Zwelling_'))
                c_sjc += 1
            elif 'Pijn' in key: 
                set_big.append(remove_prefix(key, 'Pijn_'))
                c_tjc += 1
        elif val == 1.0 and key in l_2010_small : # Disclaimer: only consider 'ACR 2010' joints
            if 'Zwelling' in key: 
                set_small.append(remove_prefix(key, 'Zwelling_'))
                c_sjc += 1
            elif 'Pijn' in key: 
                set_small.append(remove_prefix(key, 'Pijn_')) 
                c_tjc += 1
        if key not in l_2010_big and key not in l_2010_small and key not in l_ignore:
            print(key, ' is not captured by the function!')
    set_big = list(set(set_big))
    set_small = list(set(set_small))

    big_joints = len(set_big)
    small_joints = len(set_small)
    
    # correct typo
    if "bovenste spronggewicht links" in set_big: # typo!
        #print('whoop')
        set_big[set_big.index("bovenste spronggewicht links")] = "bovenste spronggewricht links"
    
    # Rename IP -> for symmetry
    row = row.rename({'Pijn_IP links' : 'Pijn_IP hand links',
                      'Pijn_IP rechts' : 'Pijn_IP hand rechts',
                        'Zwelling_IP links' : 'Zwelling_IP hand links',
                      'Zwelling_IP rechts' : 'Zwelling_IP hand rechts',
                       })
    
    # Rename PIP & DIP for symmetry
    l_pip = ['Zwelling_pip 2 links hand', 'Zwelling_pip 2 links voet', 'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 2 rechts voet', 'Zwelling_pip 3 links hand', 'Zwelling_pip 3 links voet', 'Zwelling_pip 3 rechts hand', 'Zwelling_pip 3 rechts voet',
 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 links voet', 'Zwelling_pip 4 rechts hand', 'Zwelling_pip 4 rechts voet', 'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 links voet', 'Zwelling_pip 5 rechts hand', 'Zwelling_pip 5 rechts voet', 'Pijn_pip 2 links hand', 'Pijn_pip 2 links voet', 'Pijn_pip 2 rechts hand',
 'Pijn_pip 2 rechts voet', 'Pijn_pip 3 links hand', 'Pijn_pip 3 links voet', 'Pijn_pip 3 rechts hand', 'Pijn_pip 3 rechts voet',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 links voet', 'Pijn_pip 4 rechts hand', 'Pijn_pip 4 rechts voet', 'Pijn_pip 5 links hand',
 'Pijn_pip 5 links voet', 'Pijn_pip 5 rechts hand', 'Pijn_pip 5 rechts voet', 
'Zwelling_dip 2 links','Zwelling_dip 2 links voet','Zwelling_dip 2 rechts', 'Zwelling_dip 2 rechts voet',
 'Zwelling_dip 3 links', 'Zwelling_dip 3 links voet', 'Zwelling_dip 3 rechts', 'Zwelling_dip 3 rechts voet',
 'Zwelling_dip 4 links',  'Zwelling_dip 4 links voet', 'Zwelling_dip 4 rechts', 'Zwelling_dip 4 rechts voet',
 'Zwelling_dip 5 links', 'Zwelling_dip 5 links voet', 'Zwelling_dip 5 rechts',  'Zwelling_dip 5 rechts voet', 'Pijn_dip 2 links','Pijn_dip 2 links voet', 'Pijn_dip 2 rechts', 'Pijn_dip 2 rechts voet', 
 'Pijn_dip 3 links','Pijn_dip 3 links voet', 'Pijn_dip 3 rechts', 'Pijn_dip 3 rechts voet', 
 'Pijn_dip 4 links', 'Pijn_dip 4 links voet', 'Pijn_dip 4 rechts', 'Pijn_dip 4 rechts voet', 
 'Pijn_dip 5 links', 'Pijn_dip 5 links voet', 'Pijn_dip 5 rechts', 'Pijn_dip 5 rechts voet',]
    # Loop below takes some seconds -> could probably be optimized?
    for pip in l_pip:
        new_pip = pip.split(' ')
        if len(new_pip) == 4:
            new_pip = new_pip[0] + ' ' + new_pip[3] + ' ' + new_pip[1] + ' ' + new_pip[2] 
        elif len(new_pip) == 3: # 
            new_pip = new_pip[0] + ' hand ' + new_pip[1] + ' ' + new_pip[2] 
        row = row.rename({pip : new_pip})

    # cast to lower
    set_big = [i.lower() for i in set_big] 
    set_small = [i.lower() for i in set_small] 

    # calculate nr of symmetrical joints
    d_sym = {"schouder": [0, 0], "heup": [0, 0], "knie": [0, 0], "elleboog": [0, 0], "spronggewricht": [0, 0],
            "mcp": [0, 0], "pip voet": [0, 0], "pip hand": [0, 0], "mtp": [0, 0], "ip hand":[0, 0], 
             "ip voet":[0, 0], "pols": [0, 0], "dip voet": [0, 0], "dip hand": [0, 0],
            "cervical spine": [0, 0], "sacro-ileacaal": [0, 0], "acromioclaviaculair": [0, 0], "cmc": [0, 0], 
             "sternoclaviculair": [0, 0], "manubrio sternaal gewricht": [0, 0], "tarsometatarsaal": [0, 0], "temporomandibulair": [0, 0] } # Ip voet of duim . #PIP -> hand of voet
    

    for var in set_big + set_small:
        for k in d_sym.keys():
            if k in var and ("links" in var):
                d_sym[k][0] = 1
            elif k in var and ("rechts" in var) :
                d_sym[k][1] = 1

    for var in d_sym.keys():
        if d_sym[var] == [1, 1]:
            c_sym += 1

    return big_joints, small_joints, c_sym, c_sjc, c_tjc



def process_joints(df):
    """ 
    
    In this function we disperse the big and small joints in two different datasets
    and we perform one-hot-encoding to capture all information in binary features
    
    In this function we also compute the summary statistics (for metadata):
        1. Big joint count
        2. Small joint count
        3. Symmetrical joint count
        4. Swollen joint count
        5. Tender joint count
    
    Few remarks: 
        1-2: Big/ joint distinction is made in accordance with 
             the ACR/ EULAR 2010 criteria. 
        3:   Symmetry is asserted based on an exact match for big joint, 
             but fuzzy match for the smaller joints. For example, we can 
             infer symmetry if a patient is affected in MTP-1 on the left and
             MTP-2 on the right.
        4-5: We calculated TJC and SJC ourselves because there is 
             quite some missingness otherwise.
    
    Input:
        df = the Mannequin dataset (pandas dataframe)
    Output:
        big_joints = Big joint count
        small_joints = Small joint count
        c_sym = Symmetrical joint count
        c_sjc = Swollen joint count
        c_tjc = Tender joint count
    
    """
    # Con 2010 criteria only identified small / big joints for a select view. 
    # -> we could say -> everything bigger than wrist should be included?
    
    l_2010_big = ['Pijn_schouder L', 'Pijn_schouder R', 'Pijn_Elleboog L',
             'Pijn_elleboog R', 'Pijn_heup links', 'Pijn_heup rechts', 
             'Pijn_knie links', 'Pijn_knie rechts', 
             'Zwelling_schouder L', 'Zwelling_schouder R', 'Zwelling_Elleboog L', 
             'Zwelling_elleboog R', 'Zwelling_heup links', 'Zwelling_heup rechts', 
             'Zwelling_knie links', 'Zwelling_knie rechts',
             'Pijn_onderste spronggewricht links', 'Pijn_onderste spronggewricht rechts', 'Pijn_bovenste spronggewicht links', 'Pijn_bovenste spronggewricht rechts',
             'Zwelling_onderste spronggewricht links', 'Zwelling_onderste spronggewricht rechts', 'Zwelling_bovenste spronggewicht links', 'Zwelling_bovenste spronggewricht rechts',
            'Zwelling_cervical spine', 'Pijn_cervical spine', 'Pijn_sacro-ileacaal gewricht links', 'Pijn_sacro-ileacaal gewricht rechts',
                 ] # Sacro-ileacaal has no 'Zwelling' version
    l_2010_small = ['Pijn_pols L', 'Pijn_pols R', 'Pijn_pip 2 links hand', 'Pijn_pip 2 links voet', 'Pijn_pip 2 rechts hand',
 'Pijn_pip 2 rechts voet', 'Pijn_pip 3 links hand', 'Pijn_pip 3 links voet', 'Pijn_pip 3 rechts hand', 'Pijn_pip 3 rechts voet',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 links voet', 'Pijn_pip 4 rechts hand', 'Pijn_pip 4 rechts voet', 'Pijn_pip 5 links hand',
 'Pijn_pip 5 links voet', 'Pijn_pip 5 rechts hand', 'Pijn_pip 5 rechts voet', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 
 'Pijn_mtp 2 links', 'Pijn_mtp 2 rechts', 'Pijn_mtp 3 links', 'Pijn_mtp 3 rechts', 'Pijn_mtp 4 links', 'Pijn_mtp 4 rechts',
 'Pijn_mtp 5 links', 'Pijn_mtp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts',
                   'Zwelling_pols L', 'Zwelling_pols R', 'Zwelling_pip 2 links hand', 'Zwelling_pip 2 links voet', 'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 2 rechts voet', 'Zwelling_pip 3 links hand', 'Zwelling_pip 3 links voet', 'Zwelling_pip 3 rechts hand', 'Zwelling_pip 3 rechts voet',
 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 links voet', 'Zwelling_pip 4 rechts hand', 'Zwelling_pip 4 rechts voet', 'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 links voet', 'Zwelling_pip 5 rechts hand', 'Zwelling_pip 5 rechts voet', 
 'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
 'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
 'Zwelling_mtp 2 links', 'Zwelling_mtp 2 rechts', 'Zwelling_mtp 3 links', 'Zwelling_mtp 3 rechts', 'Zwelling_mtp 4 links', 'Zwelling_mtp 4 rechts',
 'Zwelling_mtp 5 links', 'Zwelling_mtp 5 rechts', 'Zwelling_IP links', 'Zwelling_IP rechts', 
                    
                    'Pijn_IP voet links', 'Pijn_IP voet rechts', 'Zwelling_IP voet links', 'Zwelling_IP voet rechts',
                    'Pijn_cmc 1 links', 'Pijn_cmc 1 rechts','Zwelling_cmc 1 links', 'Zwelling_cmc 1 rechts',
                    'Zwelling_acromioclaviaculair L', 'Zwelling_acromioclaviaculair R', 'Pijn_acromioclaviaculair L', 'Pijn_acromioclaviaculair R',
                'Zwelling_dip 2 links','Zwelling_dip 2 links voet','Zwelling_dip 2 rechts', 'Zwelling_dip 2 rechts voet',
                 'Zwelling_dip 3 links', 'Zwelling_dip 3 links voet', 'Zwelling_dip 3 rechts', 'Zwelling_dip 3 rechts voet',
                 'Zwelling_dip 4 links',  'Zwelling_dip 4 links voet', 'Zwelling_dip 4 rechts', 'Zwelling_dip 4 rechts voet',
                 'Zwelling_dip 5 links', 'Zwelling_dip 5 links voet', 'Zwelling_dip 5 rechts',  'Zwelling_dip 5 rechts voet',
                 'Pijn_dip 2 links','Pijn_dip 2 links voet', 'Pijn_dip 2 rechts', 'Pijn_dip 2 rechts voet', 
                 'Pijn_dip 3 links','Pijn_dip 3 links voet', 'Pijn_dip 3 rechts', 'Pijn_dip 3 rechts voet', 
                 'Pijn_dip 4 links', 'Pijn_dip 4 links voet', 'Pijn_dip 4 rechts', 'Pijn_dip 4 rechts voet', 
                 'Pijn_dip 5 links', 'Pijn_dip 5 links voet', 'Pijn_dip 5 rechts', 'Pijn_dip 5 rechts voet',
                    'Zwelling_sternoclaviculair L', 'Zwelling_sternoclaviculair R', 'Pijn_sternoclaviculair L', 'Pijn_sternoclaviculair R',
                    'Zwelling_Manubrio sternaal gewricht','Pijn_Manubrio sternaal gewricht', 
                    'Zwelling_tarsometatarsaal L', 'Zwelling_tarsometatarsaal R', 'Pijn_tarsometatarsaal L', 'Pijn_tarsometatarsaal R',
                    'Zwelling_temporomandibulair L', 'Zwelling_temporomandibulair R','Pijn_temporomandibulair L', 'Pijn_temporomandibulair R',
                    'Pijn_mtp 1 links', 'Pijn_mtp 1 rechts', 'Zwelling_mtp 1 links', 'Zwelling_mtp 1 rechts',
                   ]
    
    
    l_DAS28 = ['Pijn_pols L', 'Pijn_pols R', 'Pijn_pip 2 links hand', 'Pijn_pip 2 rechts hand', 'Pijn_pip 3 links hand', 'Pijn_pip 3 rechts hand',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 rechts hand', 'Pijn_pip 5 links hand', 'Pijn_pip 5 rechts hand', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts', 'Pijn_schouder L', 'Pijn_schouder R', 'Pijn_Elleboog L','Pijn_elleboog R',
 'Pijn_knie links', 'Pijn_knie rechts', 
           
 'Zwelling_pols L', 'Zwelling_pols R', 'Zwelling_pip 2 links hand',  'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 3 links hand',  'Zwelling_pip 3 rechts hand', 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 rechts hand',  'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 rechts hand',  'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
 'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
 'Zwelling_knie links', 'Zwelling_knie rechts', 'Zwelling_schouder L', 'Zwelling_schouder R', 'Zwelling_Elleboog L',
  'Zwelling_elleboog R', 'Zwelling_IP links', 'Zwelling_IP rechts'
                   ]
    l_ignore = ['FirstConsult', 'patnr', 'pseudoId', 'Big joints', 'Small joints', 'Symmetrical joints', 'SJC', 'TJC', 'Zwelling_beiderzijds', 'Pijn_beiderzijds', 'Zwelling_ja', 'Pijn_ja'  ]
    
    
    
    df_big = df[set(df.columns).intersection(l_2010_big)].copy()
    df_small = df[set(df.columns).intersection(l_2010_small)].copy() 
    
    #print(row)
    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]
    
    #d = row.to_dict()
    set_big, set_small = [], []
    
    
    c_sjc = 0 # keep track of nr of swollen joints
    c_tjc = 0 # keep track of nr of tender joints
    
    
    def assert_symmetry(row, big=False):
        
        #c_sym = 0 # keep track of nr of symmetrical joints
        
        d_sym_reset = {"schouder": [0, 0], "heup": [0, 0], "knie": [0, 0], "elleboog": [0, 0], "spronggewricht": [0, 0],
            "mcp": [0, 0], "pip voet": [0, 0], "pip hand": [0, 0], "mtp": [0, 0], "ip hand":[0, 0], 
             "ip voet":[0, 0], "pols": [0, 0], "dip voet": [0, 0], "dip hand": [0, 0],
            "cervical spine": [0, 0], "sacro-ileacaal": [0, 0], "acromioclaviaculair": [0, 0], "cmc": [0, 0], 
             "sternoclaviculair": [0, 0], "manubrio sternaal gewricht": [0, 0], "tarsometatarsaal": [0, 0], "temporomandibulair": [0, 0] } # Ip voet of duim . #PIP -> hand of voet
        
        row.index= row.index.str.lower()
        
        # write out abbreviations
        row.index= row.index.str.replace(' r$', ' rechts', regex=True)
        row.index= row.index.str.replace(' l$', ' links', regex=True)
        new_row = row.copy()
        new_row['COUNT_SYMMETRY_%s' % ('BIG' * big + 'SMALL' * (big==False))] = 0
        new_row['COUNT_SWOLLEN_%s' % ('BIG' * big + 'SMALL' * (big==False))] = 0
        new_row['COUNT_TENDER_%s' % ('BIG' * big + 'SMALL' * (big==False))] = 0
        
        
        # Let op : Zwelling - synchroon & Pijn - synchroon
        
        # Split row in zwelling / pijn
        l_swollen = [var for var in row.index if 'zwelling' in var]
        l_tender = [var for var in row.index if 'pijn' in var]
        l_cycle = [l_swollen, l_tender]
        
        for cyc in l_cycle: 
            d_sym = d_sym_reset.copy() # reset the variable that infers symmetry
            # Initialize symmetrical swelling & painful joints
            new_row['COUNT_SYMMETRY_%s_%s' % ('TENDER' * (cyc==l_tender) + 'SWOLLEN' * (cyc==l_swollen), 'BIG' * big + 'SMALL' * (big==False))] = 0
            
            for var in cyc:
                # Count sjc / tjc
                if row[var] == 1.0:
                    if 'zwelling' in var.lower():
                        new_row['COUNT_SWOLLEN_%s' % ('BIG' * big + 'SMALL' * (big==False))] += 1
                    elif 'pijn' in var.lower():
                        new_row['COUNT_TENDER_%s' % ('BIG' * big + 'SMALL' * (big==False))] += 1

                # Count symmetry
                for k in d_sym.keys():
                    if row[var] == 1.0:
                        var = var.lower()

                        if k in var and ("links" in var):
                            d_sym[k][0] += 1 # or set it equal to 1 -> if you don't want to count double
                        elif k in var and ("rechts" in var): #  r
                            d_sym[k][1] += 1
                new_row['symmetrical_' + var.lower()] = 0 # initalize column

            # assert symmetry
            for k in d_sym.keys():
                #print(k)
                if d_sym[k][0] != 0 and d_sym[k][1] != 0: # Check if non symmetrical
                    cols = [i for i in row.index if k in i.lower()]
                    for col in cols:  # if there is symmetry -> use intial value
                        new_row['symmetrical_' +col.lower()]= row[col]

                    new_row['COUNT_SYMMETRY_%s' % ('BIG' * big + 'SMALL' * (big==False))] += d_sym[k][0] + d_sym[k][1]
                    new_row['COUNT_SYMMETRY_%s_%s' % ('TENDER' * (cyc==l_tender) + 'SWOLLEN' * (cyc==l_swollen), 'BIG' * big + 'SMALL' * (big==False))] += d_sym[k][0] + d_sym[k][1]
        
        
        # count number of big/ small joints
        new_row['COUNT_%s' % ('BIG' * big + 'SMALL' * (big==False))] = sum(row)
        return new_row
    
    
    df_big = df_big.apply(lambda x : assert_symmetry(x, big=True), axis=1) # # 
    df_small = df_small.apply(lambda x : assert_symmetry(x, big=False), axis=1) # # s

    return df_big, df_small

def count_special_joints_2(row):
    """ 
    In this function we create 5 new features:
        1. Big joint count
        2. Small joint count
        3. Symmetrical joint count
        4. Swollen joint count
        5. Tender joint count
    
    Few remarks: 
        1-2: Big/ joint distinction is made in accordance with 
             the ACR/ EULAR 2010 criteria. 
        3:   Symmetry is asserted based on an exact match for big joint, 
             but fuzzy match for the smaller joints. For example, we can 
             infer symmetry if a patient is affected in MTP-1 on the left and
             MTP-2 on the right.
        4-5: We calculated TJC and SJC ourselves because there is 
             quite some missingness otherwise.
    
    Input:
        row = row from the Mannequin dataset (pandas Series)
    Output:
        big_joints = Big joint count
        small_joints = Small joint count
        c_sym = Symmetrical joint count
        c_sjc = Swollen joint count
        c_tjc = Tender joint count
    
    """
    # Con 2010 criteria only identified small / big joints for a select view. 
    # -> we could say -> everything bigger than wrist should be included?
    
    l_2010_big = ['Pijn_schouder links', 'Pijn_schouder rechts', 'Pijn_Elleboog links',
             'Pijn_elleboog rechts', 'Pijn_heup links', 'Pijn_heup rechts', 
             'Pijn_knie links', 'Pijn_knie rechts', 
             'Zwelling_schouder links', 'Zwelling_schouder rechts', 'Zwelling_Elleboog links', 
             'Zwelling_elleboog rechts', 'Zwelling_heup links', 'Zwelling_heup rechts', 
             'Zwelling_knie links', 'Zwelling_knie rechts',
             'Pijn_onderste spronggewricht links', 'Pijn_onderste spronggewricht rechts', 'Pijn_bovenste spronggewicht links', 'Pijn_bovenste spronggewricht rechts',
             'Zwelling_onderste spronggewricht links', 'Zwelling_onderste spronggewricht rechts', 'Zwelling_bovenste spronggewicht links', 'Zwelling_bovenste spronggewricht rechts',

            'Zwelling_cervical spine', 'Pijn_cervical spine', 'Pijn_sacro-ileacaal gewricht links', 'Pijn_sacro-ileacaal gewricht rechts',
                 ] # Sacro-ileacaal has no 'Zwelling' version
    l_2010_small = ['Pijn_pols links', 'Pijn_pols rechts', 'Pijn_pip 2 links hand', 'Pijn_pip 2 links voet', 'Pijn_pip 2 rechts hand',
 'Pijn_pip 2 rechts voet', 'Pijn_pip 3 links hand', 'Pijn_pip 3 links voet', 'Pijn_pip 3 rechts hand', 'Pijn_pip 3 rechts voet',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 links voet', 'Pijn_pip 4 rechts hand', 'Pijn_pip 4 rechts voet', 'Pijn_pip 5 links hand',
 'Pijn_pip 5 links voet', 'Pijn_pip 5 rechts hand', 'Pijn_pip 5 rechts voet', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 
 'Pijn_mtp 2 links', 'Pijn_mtp 2 rechts', 'Pijn_mtp 3 links', 'Pijn_mtp 3 rechts', 'Pijn_mtp 4 links', 'Pijn_mtp 4 rechts',
 'Pijn_mtp 5 links', 'Pijn_mtp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts',
                   'Zwelling_pols links', 'Zwelling_pols rechts', 'Zwelling_pip 2 links hand', 'Zwelling_pip 2 links voet', 'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 2 rechts voet', 'Zwelling_pip 3 links hand', 'Zwelling_pip 3 links voet', 'Zwelling_pip 3 rechts hand', 'Zwelling_pip 3 rechts voet',
 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 links voet', 'Zwelling_pip 4 rechts hand', 'Zwelling_pip 4 rechts voet', 'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 links voet', 'Zwelling_pip 5 rechts hand', 'Zwelling_pip 5 rechts voet', 
 'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
 'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
 'Zwelling_mtp 2 links', 'Zwelling_mtp 2 rechts', 'Zwelling_mtp 3 links', 'Zwelling_mtp 3 rechts', 'Zwelling_mtp 4 links', 'Zwelling_mtp 4 rechts',
 'Zwelling_mtp 5 links', 'Zwelling_mtp 5 rechts', 'Zwelling_IP links', 'Zwelling_IP rechts', 
                    
                    'Pijn_IP voet links', 'Pijn_IP voet rechts', 'Zwelling_IP voet links', 'Zwelling_IP voet rechts',
                    'Pijn_cmc 1 links', 'Pijn_cmc 1 rechts','Zwelling_cmc 1 links', 'Zwelling_cmc 1 rechts',
                    'Zwelling_acromioclaviaculair links', 'Zwelling_acromioclaviaculair rechts', 'Pijn_acromioclaviaculair links', 'Pijn_acromioclaviaculair rechts',
                'Zwelling_dip 2 links','Zwelling_dip 2 links voet','Zwelling_dip 2 rechts', 'Zwelling_dip 2 rechts voet',
                 'Zwelling_dip 3 links', 'Zwelling_dip 3 links voet', 'Zwelling_dip 3 rechts', 'Zwelling_dip 3 rechts voet',
                 'Zwelling_dip 4 links',  'Zwelling_dip 4 links voet', 'Zwelling_dip 4 rechts', 'Zwelling_dip 4 rechts voet',
                 'Zwelling_dip 5 links', 'Zwelling_dip 5 links voet', 'Zwelling_dip 5 rechts',  'Zwelling_dip 5 rechts voet',
                 'Pijn_dip 2 links','Pijn_dip 2 links voet', 'Pijn_dip 2 rechts', 'Pijn_dip 2 rechts voet', 
                 'Pijn_dip 3 links','Pijn_dip 3 links voet', 'Pijn_dip 3 rechts', 'Pijn_dip 3 rechts voet', 
                 'Pijn_dip 4 links', 'Pijn_dip 4 links voet', 'Pijn_dip 4 rechts', 'Pijn_dip 4 rechts voet', 
                 'Pijn_dip 5 links', 'Pijn_dip 5 links voet', 'Pijn_dip 5 rechts', 'Pijn_dip 5 rechts voet',
                    'Zwelling_sternoclaviculair links', 'Zwelling_sternoclaviculair rechts', 'Pijn_sternoclaviculair links', 'Pijn_sternoclaviculair rechts',
                    'Zwelling_Manubrio sternaal gewricht','Pijn_Manubrio sternaal gewricht', 
                    'Zwelling_tarsometatarsaal links', 'Zwelling_tarsometatarsaal rechts', 'Pijn_tarsometatarsaal links', 'Pijn_tarsometatarsaal rechts',
                    'Zwelling_temporomandibulair links', 'Zwelling_temporomandibulair rechts','Pijn_temporomandibulair links', 'Pijn_temporomandibulair rechts',
                    'Pijn_mtp 1 links', 'Pijn_mtp 1 rechts', 'Zwelling_mtp 1 links', 'Zwelling_mtp 1 rechts',
                   ]
    l_DAS28 = ['Pijn_pols links', 'Pijn_pols rechts', 'Pijn_pip 2 links hand', 'Pijn_pip 2 rechts hand', 'Pijn_pip 3 links hand', 'Pijn_pip 3 rechts hand',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 rechts hand', 'Pijn_pip 5 links hand', 'Pijn_pip 5 rechts hand', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts', 'Pijn_schouder links', 'Pijn_schouder rechts', 'Pijn_Elleboog links','Pijn_elleboog rechts',
 'Pijn_knie links', 'Pijn_knie rechts', 
           
 'Zwelling_pols links', 'Zwelling_pols rechts', 'Zwelling_pip 2 links hand',  'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 3 links hand',  'Zwelling_pip 3 rechts hand', 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 rechts hand',  'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 rechts hand',  'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
 'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
 'Zwelling_knie links', 'Zwelling_knie rechts', 'Zwelling_schouder links', 'Zwelling_schouder rechts', 'Zwelling_Elleboog links',
  'Zwelling_elleboog rechts', 'Zwelling_IP links', 'Zwelling_IP rechts'
                   ]
    l_ignore = ['FirstConsult', 'patnr', 'pseudoId', 'Big joints', 'Small joints', 'Symmetrical joints', 'SJC', 'TJC', 'Zwelling_beiderzijds', 'Pijn_beiderzijds', 'Zwelling_ja', 'Pijn_ja'  ]
    
    #print(row)
    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]
    
    #row.index= row.index.str.lower()
        
    # write out abbreviations
    row.index= row.index.str.replace(' R$', ' rechts', regex=True)
    row.index= row.index.str.replace(' L$', ' links', regex=True)
    
    d = row.to_dict()
    set_big, set_small = [], []
    
    
    c_sjc = 0 # keep track of nr of swollen joints
    c_tjc = 0 # keep track of nr of tender joints
    c_sym = 0 # keep track of nr of symmetrical joints
    
    for key in d.keys(): # This complicated function is needed to ensure that tender/swollen joints aren't counted twice
        val = d[key]
        if val == 1.0 and key in l_2010_big : # Disclaimer: only consider 'ACR 2010' joints
            if 'Zwelling' in key: 
                set_big.append(remove_prefix(key, 'Zwelling_'))
                c_sjc += 1
            elif 'Pijn' in key: 
                set_big.append(remove_prefix(key, 'Pijn_'))
                c_tjc += 1
        elif val == 1.0 and key in l_2010_small : # Disclaimer: only consider 'ACR 2010' joints
            if 'Zwelling' in key: 
                set_small.append(remove_prefix(key, 'Zwelling_'))
                c_sjc += 1
            elif 'Pijn' in key: 
                set_small.append(remove_prefix(key, 'Pijn_')) 
                c_tjc += 1
        if key not in l_2010_big and key not in l_2010_small and key not in l_ignore:
            print(key, ' is not captured by the function!')
    set_big = list(set(set_big))
    set_small = list(set(set_small))

    big_joints = len(set_big)
    small_joints = len(set_small)
    
    # correct typo
    if "bovenste spronggewicht links" in set_big: # typo!
        #print('whoop')
        set_big[set_big.index("bovenste spronggewicht links")] = "bovenste spronggewricht links"
    
    # Rename IP -> for symmetry
    row = row.rename({'Pijn_IP links' : 'Pijn_IP hand links',
                      'Pijn_IP rechts' : 'Pijn_IP hand rechts',
                        'Zwelling_IP links' : 'Zwelling_IP hand links',
                      'Zwelling_IP rechts' : 'Zwelling_IP hand rechts',
                       })
    
    # Rename PIP & DIP for symmetry
    l_pip = ['Zwelling_pip 2 links hand', 'Zwelling_pip 2 links voet', 'Zwelling_pip 2 rechts hand',
 'Zwelling_pip 2 rechts voet', 'Zwelling_pip 3 links hand', 'Zwelling_pip 3 links voet', 'Zwelling_pip 3 rechts hand', 'Zwelling_pip 3 rechts voet',
 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 links voet', 'Zwelling_pip 4 rechts hand', 'Zwelling_pip 4 rechts voet', 'Zwelling_pip 5 links hand',
 'Zwelling_pip 5 links voet', 'Zwelling_pip 5 rechts hand', 'Zwelling_pip 5 rechts voet', 'Pijn_pip 2 links hand', 'Pijn_pip 2 links voet', 'Pijn_pip 2 rechts hand',
 'Pijn_pip 2 rechts voet', 'Pijn_pip 3 links hand', 'Pijn_pip 3 links voet', 'Pijn_pip 3 rechts hand', 'Pijn_pip 3 rechts voet',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 links voet', 'Pijn_pip 4 rechts hand', 'Pijn_pip 4 rechts voet', 'Pijn_pip 5 links hand',
 'Pijn_pip 5 links voet', 'Pijn_pip 5 rechts hand', 'Pijn_pip 5 rechts voet', 
'Zwelling_dip 2 links','Zwelling_dip 2 links voet','Zwelling_dip 2 rechts', 'Zwelling_dip 2 rechts voet',
 'Zwelling_dip 3 links', 'Zwelling_dip 3 links voet', 'Zwelling_dip 3 rechts', 'Zwelling_dip 3 rechts voet',
 'Zwelling_dip 4 links',  'Zwelling_dip 4 links voet', 'Zwelling_dip 4 rechts', 'Zwelling_dip 4 rechts voet',
 'Zwelling_dip 5 links', 'Zwelling_dip 5 links voet', 'Zwelling_dip 5 rechts',  'Zwelling_dip 5 rechts voet', 'Pijn_dip 2 links','Pijn_dip 2 links voet', 'Pijn_dip 2 rechts', 'Pijn_dip 2 rechts voet', 
 'Pijn_dip 3 links','Pijn_dip 3 links voet', 'Pijn_dip 3 rechts', 'Pijn_dip 3 rechts voet', 
 'Pijn_dip 4 links', 'Pijn_dip 4 links voet', 'Pijn_dip 4 rechts', 'Pijn_dip 4 rechts voet', 
 'Pijn_dip 5 links', 'Pijn_dip 5 links voet', 'Pijn_dip 5 rechts', 'Pijn_dip 5 rechts voet',]
    # Loop below takes some seconds -> could probably be optimized?
    for pip in l_pip:
        new_pip = pip.split(' ')
        if len(new_pip) == 4:
            new_pip = new_pip[0] + ' ' + new_pip[3] + ' ' + new_pip[1] + ' ' + new_pip[2] 
        elif len(new_pip) == 3: # 
            new_pip = new_pip[0] + ' hand ' + new_pip[1] + ' ' + new_pip[2] 
        row = row.rename({pip : new_pip})

    # cast to lower
    set_big = [i.lower() for i in set_big] 
    set_small = [i.lower() for i in set_small] 

    # calculate nr of symmetrical joints
    d_sym = {"schouder": [0, 0], "heup": [0, 0], "knie": [0, 0], "elleboog": [0, 0], "spronggewricht": [0, 0],
            "mcp": [0, 0], "pip voet": [0, 0], "pip hand": [0, 0], "mtp": [0, 0], "ip hand":[0, 0], 
             "ip voet":[0, 0], "pols": [0, 0], "dip voet": [0, 0], "dip hand": [0, 0],
            "cervical spine": [0, 0], "sacro-ileacaal": [0, 0], "acromioclaviaculair": [0, 0], "cmc": [0, 0], 
             "sternoclaviculair": [0, 0], "manubrio sternaal gewricht": [0, 0], "tarsometatarsaal": [0, 0], "temporomandibulair": [0, 0] } # Ip voet of duim . #PIP -> hand of voet
    

    for var in set_big + set_small:
        for k in d_sym.keys():
            if k in var and ("links" in var):
                d_sym[k][0] = 1
            elif k in var and ("rechts" in var) :
                d_sym[k][1] = 1

    for var in d_sym.keys():
        if d_sym[var] == [1, 1]:
            c_sym += 1

    return big_joints, small_joints, c_sym, c_sjc, c_tjc

def rename_mannequin_features(feature):
    """
    Translate dutch feature names to english equivalent
    
    Input:
        feature = old feature name
        new = new feature name
    
    """
    new = feature
    if 'Zwelling' in feature:
        new = 'Swollen '
    elif 'Pijn' in feature :
        new = 'Tender '
    else : 
        if feature == 'BSE':
            new = 'ESR'
        if feature == 'Hb':
            new = 'Hemoglobin'
        if feature == 'Ht':
            new = 'Hematocrit'
        if feature == 'Trom':
            new = 'Thrombocytes'
        if feature == 'Lym':
            new = 'Lymphocytes'
        if feature == 'Leuko':
            new = 'Leukocytes'   
    
    if 'elleboog' in feature or 'Elleboog' in feature: # Elbow
        new += 'elbow'
    if 'IP' in feature and 'voet' not in feature: # IP hand
        new += 'IP hand' # interphalangeal
    if 'IP' in feature and 'voet' in feature: # IP feet 
        new += 'IP foot' # interphalangeal
    if 'acromioclaviaculair' in feature : # Acromioclaviculair
        new += 'acromioclaviculair'
    if 'bovenste sprong' in feature : # Ankle -> We cap it at "sprong..." (because "gewricht" is sometimes misspelled)
        new += 'ankle'
    if 'onderste sprong' in feature : # talo-calcaneo (closest to big toe!)
        new += 'talo-calcaneo-navicularis'
    if 'tarsometatarsaal' in feature : # Tarso (furthest removed from big toe!)
        new += 'tarsometatarsal'
    if 'cmc' in feature : # Carpometacarpal 1 (below thumb joint)
        new += 'CMC' # 
    if 'dip 2' in feature and 'voet' not in feature: # dip 2 top hand
        new +=  'DIP 2 hand' # 'Distal Interphalangeal'
    if 'dip 3' in feature and 'voet' not in feature: # dip 3 top hand
        new +=  'DIP 3 hand' # 'Distal Interphalangeal'
    if 'dip 4' in feature and 'voet' not in feature: # dip 4 top hand
        new +=  'DIP 4 hand' # 'Distal Interphalangeal
    if 'dip 5' in feature and 'voet' not in feature: # dip 5 top hand
        new +=  'DIP 5 hand' # 'Distal Interphalangeal
    if 'dip 2' in feature and 'voet' in feature: # dip 2 top feet
        new +=  'DIP 2 foot' # 'Distal Interphalangeal
    if 'dip 3' in feature and 'voet' in feature: # dip 3 top feet
        new +=  'DIP 3 foot' # 'Distal Interphalangeal
    if 'dip 4' in feature and 'voet' in feature: # dip 4 top feet
        new +=  'DIP 4 foot' # 'Distal Interphalangeal
    if 'dip 5' in feature and 'voet' in feature: # dip 5 top feet
        new +=  'DIP 5 foot' # 'Distal Interphalangeal
        
    if 'pip 2' in feature and 'voet' not in feature: # pip 2 middle hand
        new += 'PIP 2 hand' # Proximal Interphalangeal joints
    if 'pip 3' in feature and 'voet' not in feature: # pip 3 middle hand
        new += 'PIP 3 hand' # Proximal Interphalangeal joints
    if 'pip 4' in feature and 'voet' not in feature: # pip 4 middle hand
        new += 'PIP 4 hand' # Proximal Interphalangeal joints
    if 'pip 5' in feature and 'voet' not in feature: # pip 5 middle hand
        new += 'PIP 5 hand' # Proximal Interphalangeal joints
        
    if 'pip 2' in feature and 'voet' in feature: # pip 2 middle foot
        new += 'PIP 2 foot' # Proximal Interphalangeal joints
    if 'pip 3' in feature and 'voet' in feature: # pip 3 middle foot
        new += 'PIP 3 foot' # Proximal Interphalangeal joints
    if 'pip 4' in feature and 'voet' in feature: # pip 4 middle foot
        new += 'PIP 4 foot' # Proximal Interphalangeal joints
    if 'pip 5' in feature and 'voet' in feature: # pip 5 middle foot
        new += 'PIP 5 foot' # Proximal Interphalangeal joints
    
    if 'mcp 1' in feature : # mcp 1 thumb (middle)
        new += 'MCP 1'
    if 'mcp 2' in feature : # mcp 2 lower hand
        new += 'MCP 2'
    if 'mcp 3' in feature : # mcp 3 lower hand
        new += 'MCP 3'
    if 'mcp 4' in feature : # mcp 4 lower hand
        new += 'MCP 4'
    if 'mcp 5' in feature : # mcp 5 lower hand
        new += 'MCP 5'
        
    if 'mtp 1' in feature : # mtp 1 thumb (middle)
        new += 'MTP 1'
    if 'mtp 2' in feature : # mtp 2 lower hand
        new += 'MTP 2'
    if 'mtp 3' in feature : # mtp 3 lower hand
        new += 'MTP 3'
    if 'mtp 4' in feature : # mtp 4 lower hand
        new += 'MTP 4'
    if 'mtp 5' in feature : # mtp 5 lower hand
        new += 'MTP 5'
        
    if 'pols' in feature :  # Wrist joint
        new += 'wrist'
    if 'heup' in feature : # Hip joint
        new += 'hip'
    if 'knie' in feature : # Hip joint
        new += 'knee'
    if 'schouder' in feature : # Shoulder
        new += 'shoulder'
    if 'sternoclaviculair' in feature : # Sternoclaviculair (breast)
        new+= 'sternoclavicular'
    if 'sacro-ileacaal' in feature : # Sternoclaviculair (breast)
        new+= 'sacroiliac'
    if 'temporomandibulair' in feature: # temporomandibular  (Jaw)
        new+= 'temporomandibular'
    
    
    # No left or right
    if 'Manubrio' in feature: 
        new+= 'manubriosternal'
    if 'cervical spine' in feature:
        new+= 'cervical spine'
    if 'rechts' in feature or  re.search('R$', feature): 
        new += ' R'
    elif 'links' in feature or  re.search('L$', feature): 
        new += ' L'
    return new


def get_mannequin_coord(feature):
    # Coordinates are based on figure:
    # RA_Clustering/figures/2_processing/Mannequin_large_old.jpg
    
    x,y,s = 0, 0, 80
    
    if 'rechts' in feature or 'R' in feature :
        if 'elleboog' in feature: # Elbow
            x, y = 543, 584
            s=130
        if 'IP' in feature and 'voet' not in feature: # IP hand
            x, y = 409, 1148
        if 'IP' in feature and 'voet' in feature: # IP feet
            x, y = 566, 1677
        if 'acromioclaviaculair' in feature : # Acromioclaviculair
            x, y = 594, 314
            s = 30
        
        if 'tarsometatarsaal' in feature : # Tarso (furthest removed from big toe!)
            x, y = 576, 1466
            s = 40     
        
        if 'cmc' in feature : # CMC 1 (below thumb joint)
            x, y = 515, 986
            s = 30
            
        if 'dip 2' in feature and 'voet' not in feature: # dip 2 top hand
            x, y = 255, 1230
        if 'dip 3' in feature and 'voet' not in feature: # dip 3 top hand
            x, y = 152, 1168
        if 'dip 4' in feature and 'voet' not in feature: # dip 4 top hand
            x, y = 99, 1101
        if 'dip 5' in feature and 'voet' not in feature: # dip 5 top hand
            x, y = 96, 1005
            
        if 'dip 2' in feature and 'voet' in feature: # dip 2 top feet
            x, y = 421, 1706
            s = 60
        if 'dip 3' in feature and 'voet' in feature: # dip 3 top feet
            x, y = 339, 1661
            s = 60
        if 'dip 4' in feature and 'voet' in feature: # dip 4 top feet
            x, y = 266, 1608
            s = 60
        if 'dip 5' in feature and 'voet' in feature: # dip 5 top feet
            x, y = 212, 1545
            s = 60
            
        if 'pip 2' in feature and 'voet' not in feature: # pip 2 middle hand
            x, y = 334, 1127
        if 'pip 3' in feature and 'voet' not in feature: # pip 3 middle hand
            x, y = 236, 1083
        if 'pip 4' in feature and 'voet' not in feature: # pip 4 middle hand
            x, y = 172, 1018
        if 'pip 5' in feature and 'voet' not in feature: # pip 5 middle hand
            x, y = 141, 932
            
        if 'pip 2' in feature and 'voet' in feature: # pip 2 middle feet
            x, y = 471, 1645
        if 'pip 3' in feature and 'voet' in feature: # pip 3 middle feet
            x, y = 388, 1600
        if 'pip 4' in feature and 'voet' in feature: # pip 4 middle feet
            x, y = 320, 1550
        if 'pip 5' in feature and 'voet' in feature: # pip 5 middle feet
            x, y = 267, 1489
            
        if 'mcp 1' in feature : # mcp 1 thumb (middle)
            x, y = 467, 1051
        if 'mcp 2' in feature : # mcp 2 lower hand
            x, y = 418, 997
        if 'mcp 3' in feature : # mcp 3 lower hand
            x, y = 346, 961
        if 'mcp 4' in feature : # mcp 4 lower hand
            x, y = 291, 904
        if 'mcp 5' in feature : # mcp 5 lower hand
            x, y = 234, 848
            
        if 'mtp 1' in feature : # mtp 1 lower feet (big toe)
            x, y = 606, 1603
        if 'mtp 2' in feature : # mtp 2 lower feet
            x, y = 524, 1590
        if 'mtp 3' in feature : # mtp 3 lower feet
            x, y = 454, 1554
        if 'mtp 4' in feature : # mtp 4 lower feet
            x, y = 388, 1505
        if 'mtp 5' in feature : # mtp 5 lower feet
            x, y = 338, 1444
        
        if 'pols' in feature : # Wrist joint
            x, y = 498, 779
            s = 110
        if 'heup' in feature : # Hip joint
            x, y = 629, 791
        if 'knie' in feature : # Knee
            x, y = 653, 1117
            s = 110
        if 'schouder' in feature : # Shoulder
            x, y = 552, 361
            s = 110
        
        if 'bovenste sprong' in feature : # Ankle
            x, y = 644, 1425
            s = 110
        if 'onderste sprong' in feature : # talo-calcaneo (closest to big toe!)
            x, y = 629, 1494
            s = 40
        if 'sternoclaviculair' in feature : # Sternoclaviculair (breast)
            x, y = 675, 320
            s = 50
            
        if 'sacro-ileacaal' in feature : # SI-joint (right below cervical spine)
            x, y = 677, 680
            s = 30
            
        if 'temporomandibulair' in feature: # Jaw 
            x, y = 652, 210
    
    # 'links' in feature or 'L' in feature        
    if 'links' in feature or 'L' in feature : 
        if 'Elleboog' in feature: # Elbow
            x, y = 914, 550
            s=130
        if 'IP' in feature and 'voet' not in feature: # IP hand
            x, y = 1047, 1076
        if 'IP' in feature and 'voet' in feature: # IP feet 
            x, y = 816, 1697
        if 'acromioclaviaculair' in feature : # Acromioclaviculair (close to shoulder)
            x, y = 846, 296
            s = 30
        if 'bovenste sprong' in feature : # Ankle
            x, y = 808, 1447
            s = 110
        if 'onderste sprong' in feature : # talo-calcaneo (closest to big toe!)
            x, y = 819, 1515
            s = 40
        if 'tarsometatarsaal' in feature : # Tarso (furthest removed from big toe!)
            x, y = 865, 1490
            s = 40    
        
        if 'cmc' in feature : # CMC 1 (below thumb joint)
            x, y = 939, 905
            s = 30
        if 'dip 2' in feature and 'voet' not in feature: # dip 2 top hand
            x, y = 1210, 1132
        if 'dip 3' in feature and 'voet' not in feature: # dip 3 top hand
            x, y = 1303, 1059
        if 'dip 4' in feature and 'voet' not in feature: # dip 4 top hand
            x, y = 1346, 980
        if 'dip 5' in feature and 'voet' not in feature: # dip 5 top hand
            x, y = 1352, 875
            
        if 'dip 2' in feature and 'voet' in feature: # dip 2 top feet
            x, y = 936, 1733
            s = 60
        if 'dip 3' in feature and 'voet' in feature: # dip 3 top feet
            x, y = 1013, 1700
            s = 60
        if 'dip 4' in feature and 'voet' in feature: # dip 4 top feet
            x, y = 1080, 1638
            s = 60
        if 'dip 5' in feature and 'voet' in feature: # dip 5 top feet
            x, y = 1143, 1576
            s = 60
            
        if 'pip 2' in feature and 'voet' not in feature: # pip 2 middle hand
            x, y = 1114, 1053
        if 'pip 3' in feature and 'voet' not in feature: # pip 3 middle hand
            x, y = 1215, 985
        if 'pip 4' in feature and 'voet' not in feature: # pip 4 middle hand
            x, y = 1270, 911
        if 'pip 5' in feature and 'voet' not in feature: # pip 5 middle hand
            x, y = 1292, 821
            
        if 'pip 2' in feature and 'voet' in feature: # pip 2 middle feet
            x, y = 897, 1670
        if 'pip 3' in feature and 'voet' in feature: # pip 3 middle feet
            x, y = 969, 1635
        if 'pip 4' in feature and 'voet' in feature: # pip 4 middle feet
            x, y = 1032, 1574
        if 'pip 5' in feature and 'voet' in feature: # pip 5 middle feet
            x, y = 1094, 1514
        
        if 'mcp 1' in feature : # mcp 1 thumb (middle)
            x, y = 978, 988
        if 'mcp 2' in feature : # mcp 2 lower hand
            x, y = 1015, 918
        if 'mcp 3' in feature : # mcp 3 lower hand
            x, y = 1086, 877
        if 'mcp 4' in feature : # mcp 4 lower hand
            x, y = 1134, 812
        if 'mcp 5' in feature : # mcp 5 lower hand
            x, y = 1182, 749
            
        if 'mtp 1' in feature : # mtp 1 lower feet (big toe)
            x, y = 779, 1624
        if 'mtp 2' in feature : # mtp 2 lower hand
            x, y = 855, 1600
        if 'mtp 3' in feature : # mtp 3 lower hand
            x, y = 928, 1570
        if 'mtp 4' in feature : # mtp 4 lower hand
            x, y = 984, 1512
        if 'mtp 5' in feature : # mtp 5 lower hand
            x, y = 1039, 1457
        
        if 'pols' in feature : # Wrist joint
            x, y = 986, 705
            s = 110
        if 'heup' in feature : # Hip joint
            x, y = 799, 783
        if 'knie' in feature : # Knee
            x, y = 815, 1112
            s = 110
        if 'schouder' in feature : # Shoulder
            x, y = 880, 345
            s = 110
        
        if 'sternoclaviculair' in feature : # Sternoclaviculair (breast)
            x, y = 756, 316
            s = 50
          
        if 'sacro-ileacaal' in feature : # SI-joint (right below cervical spine)
            x, y = 748, 680
            s = 30
        
        if 'temporomandibulair' in feature: # Jaw 
            x, y = 772, 194
    
    # No left or right
    if 'Manubrio' in feature: 
        x, y = 711,381 # ?
    if 'cervical spine' in feature:
        x, y = 707, 280 # ?
        s = 30
      
    return x, y, s

def get_mannequin_coord_lowerRes(feature):
    # Coordinates are based on figure:
    # RA_Clustering/figures/2_processing/Mannequin_large_old.jpg
    
    x,y,s = 0, 0, 80
    
    if 'rechts' in feature or 'R' in feature :
        if 'Elbow' in feature: # Elbow
            x, y = 543, 584
            s=130
        if 'IP' in feature and 'feet' not in feature: # IP hand
            x, y = 409, 1148
        if 'IP' in feature and 'voet' in feature: # IP feet
            x, y = 566, 1677

        if 'PIP' in feature and 'Feet' not in feature: # pip 3 middle hand
            x, y = 236, 1083
            s=300
            
        if 'MCP' in feature : # mcp 3 lower hand
            x, y = 346, 961
            s=300
        if 'MTP' in feature : # any mtp
            x, y = 462, 1540
            s=300
        if 'Wrist' in feature : # Wrist joint
            x, y = 498, 779
            s = 110
        if 'Hip' in feature : # Hip joint
            x, y = 629, 791
        if 'Knee' in feature : # Knee
            x, y = 653, 1117
            s = 110
        if 'Shoulder' in feature : # Shoulder
            x, y = 552, 361
            s = 110
        if 'Ankle' in feature : # Ankle
            x, y = 644, 1425
            s = 110
    
    # 'links' in feature or 'L' in feature        
    if 'links' in feature or 'L' in feature : 
        if 'Elbow' in feature: # Elbow
            x, y = 914, 550
            s=130
        if 'IP' in feature and 'PIP' not in feature and 'feet' not in feature: # IP hand
            x, y = 1047, 1076
        if 'Ankle' in feature : # Ankle
            x, y = 808, 1447
            s = 110
        
        if 'PIP' in feature and 'feet' not in feature: # pip 3 middle hand
            x, y = 1215, 985
            s=300
        
        if 'MCP' in feature : # mcp 3 lower hand
            x, y = 1086, 877
            s=300
        if 'MTP' in feature : # mtp 3 lower hand
            x, y = 935, 1560
            s=300
        
        if 'Wrist' in feature : # Wrist joint
            x, y = 986, 705
            s = 110
        if 'Hip' in feature : # Hip joint
            x, y = 799, 783
        if 'Knee' in feature : # Knee
            x, y = 815, 1112
            s = 110
        if 'Shoulder' in feature : # Shoulder
            x, y = 880, 345
            s = 110
      
    return x, y, s

def get_mannequin_coord_lowRes(feature):
    # Coordinates are based on figure:
    # RA_Clustering/figures/2_processing/Mannequin_large_old.jpg
    
    x,y,s = 0, 0, 80
    
    if 'rechts' in feature or 'R' in feature :
        if 'Elbow' in feature: # Elbow
            x, y = 543, 584
            s=130
        if 'PIP_L1' in feature and 'feet' not in feature: # IP hand
            x, y = 409, 1148
        if 'PIP_L2' in feature and 'voet' not in feature: # pip 2 middle hand
            x, y = 334, 1127
        if 'PIP_L3' in feature and 'voet' not in feature: # pip 3 middle hand
            x, y = 236, 1083
        if 'PIP_L4' in feature and 'voet' not in feature: # pip 4 middle hand
            x, y = 172, 1018
        if 'PIP_L5' in feature and 'voet' not in feature: # pip 5 middle hand
            x, y = 141, 932
            
        if 'MCP_L1' in feature : # mcp 1 thumb (middle)
            x, y = 467, 1051
        if 'MCP_L2' in feature : # mcp 2 lower hand
            x, y = 418, 997
        if 'MCP_L3' in feature : # mcp 3 lower hand
            x, y = 346, 961
        if 'MCP_L4' in feature : # mcp 4 lower hand
            x, y = 291, 904
        if 'MCP_L5' in feature : # mcp 5 lower hand
            x, y = 234, 848
            
        if 'MTP_L1' in feature : # mtp 1 lower feet (big toe)
            x, y = 606, 1603
        if 'MTP_L2' in feature : # mtp 2 lower feet
            x, y = 524, 1590
        if 'MTP_L3' in feature : # mtp 3 lower feet
            x, y = 454, 1554
        if 'MTP_L4' in feature : # mtp 4 lower feet
            x, y = 388, 1505
        if 'MTP_L5' in feature : # mtp 5 lower feet
            x, y = 338, 1444

        
        if 'Wrist' in feature : # Wrist joint
            x, y = 498, 779
            s = 110
        if 'Hip' in feature : # Hip joint
            x, y = 629, 791
        if 'Knee' in feature : # Knee
            x, y = 653, 1117
            s = 110
        if 'Shoulder' in feature : # Shoulder
            x, y = 552, 361
            s = 110
        if 'Ankle' in feature : # Ankle
            x, y = 644, 1425
            s = 110
    
    # 'links' in feature or 'L' in feature        
    if 'links' in feature or 'L' in feature : 
        if 'Elbow' in feature: # Elbow
            x, y = 914, 550
            s=130
        
        if 'Ankle' in feature : # Ankle
            x, y = 808, 1447
            s = 110
            
        if 'PIP_R1' in feature and 'feet' not in feature: # IP hand
            x, y = 1047, 1076
        if 'PIP_R2' in feature and 'voet' not in feature: # pip 2 middle hand
            x, y = 1114, 1053
        if 'PIP_R3' in feature and 'voet' not in feature: # pip 3 middle hand
            x, y = 1215, 985
        if 'PIP_R4' in feature and 'voet' not in feature: # pip 4 middle hand
            x, y = 1270, 911
        if 'PIP_R5' in feature and 'voet' not in feature: # pip 5 middle hand
            x, y = 1292, 821
            
        if 'MCP_R1' in feature : # mcp 1 thumb (middle)
            x, y = 978, 988
        if 'MCP_R2' in feature : # mcp 2 lower hand
            x, y = 1015, 918
        if 'MCP_R3' in feature : # mcp 3 lower hand
            x, y = 1086, 877
        if 'MCP_R4' in feature : # mcp 4 lower hand
            x, y = 1134, 812
        if 'MCP_R5' in feature : # mcp 5 lower hand
            x, y = 1182, 749
            
        if 'MTP_R1' in feature : # mtp 1 lower feet (big toe)
            x, y = 779, 1624
        if 'MTP_R2' in feature : # mtp 2 lower feet
            x, y = 855, 1600
        if 'MTP_R3' in feature : # mtp 3 lower feet
            x, y = 928, 1570
        if 'MTP_R4' in feature : # mtp 4 lower feet
            x, y = 984, 1512
        if 'MTP_R5' in feature : # mtp 5 lower feet
            x, y = 1039, 1457
 
        
        if 'Wrist' in feature : # Wrist joint
            x, y = 986, 705
            s = 110
        if 'Hip' in feature : # Hip joint
            x, y = 799, 783
        if 'Knee' in feature : # Knee
            x, y = 815, 1112
            s = 110
        if 'Shoulder' in feature : # Shoulder
            x, y = 880, 345
            s = 110

      
    return x, y, s