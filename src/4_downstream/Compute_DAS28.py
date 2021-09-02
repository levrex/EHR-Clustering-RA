import pandas as pd
import numpy as np
import re
import numpy as np

# Import Mannequin data with Sedimentation rate (BSE)
new_df = pd.read_csv(r'data/1_raw/Clustering_Gewrichtspop_with_BSE.csv', sep='|')
new_df = new_df.sort_values(by=['PATNR', 'DATUM'])

# Get patient_ids
df_pat = pd.read_csv(r'filters/RA_patients_083.csv', sep=',', index_col=0)
new_pat = dict(zip(df_pat.PATNR, df_pat.PEC))

import pandas as pd
import numpy as np
import re

def compute_DAS(df, patnr=8129908, first_date=None, verbose=False):
    """ 
    Subset on patnr and date (snapshot) and determine the number of 
    swollen- and tender joints. 
    
    We apply a fuzzy matching. The ESR is regularly out of sync with 
    the mannequin joint information, hence we apply a time window 
    (2 weeks before the visit and max of 2 days after the visit).
    
    We transform the dates into days according to the first date 
    provided by the user.
    
    We only calculate the DAS28(3) if we have joint information
    and the ESR variable available. We perform simple checks
    to ensure we have the joint information necessary. 
    They ensure us that we are not missing any information:
        - no_pain = indicates that there are no tender joints
        - no_swollen = indicates that there are no swollen joints
            
    If both of these variables score positive then we can safely
    compute DAS28(3) with only the ESR. However, we cease the DAS computation
    if we don't find any joint information while these variables 
    indicate the opposite, or when these variables aren't defined at all 
    (because of missing information).
    
    Input:
        - df = pandas Dataframe with mannequin data as well as ESR 
        - patnr = patient id
        - first date = users may supply the first consult date
            - this function takes the first date in the dataframe (df) by default
        - verbose = print out all information related to DAS computation
            - Indicate whenever the ESR is imputed (observation carried forward or backward)
            - Indicate whenever a DAS is extracted directly from the table (i.e. not computed)
            - Indicate whenever the DAS could not be computed because joint information is missing
            - Indicate whenever the DAS could not be computed because ESR is missing
            
    ToDo: get BSE from LAB
    """
    l_DAS28_tjc = ['Pijn_pols L', 'Pijn_pols R', 'Pijn_pip 2 links hand', 'Pijn_pip 2 rechts hand', 'Pijn_pip 3 links hand', 'Pijn_pip 3 rechts hand',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 rechts hand', 'Pijn_pip 5 links hand', 'Pijn_pip 5 rechts hand', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts', 'Pijn_schouder L', 'Pijn_schouder R', 'Pijn_Elleboog L','Pijn_elleboog R',
 'Pijn_knie links', 'Pijn_knie rechts']
           
    l_DAS28_sjc = [ 'Zwelling_pols L', 'Zwelling_pols R', 'Zwelling_pip 2 links hand',  'Zwelling_pip 2 rechts hand',
     'Zwelling_pip 3 links hand',  'Zwelling_pip 3 rechts hand', 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 rechts hand',  'Zwelling_pip 5 links hand',
     'Zwelling_pip 5 rechts hand',  'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
     'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
     'Zwelling_knie links', 'Zwelling_knie rechts', 'Zwelling_schouder L', 'Zwelling_schouder R', 'Zwelling_Elleboog L',
      'Zwelling_elleboog R', 'Zwelling_IP links', 'Zwelling_IP rechts'
                       ]
     
    l_DAS44_tjc = ['Pijn_pols L', 'Pijn_pols R', 'Pijn_pip 2 links hand', 'Pijn_pip 2 rechts hand', 'Pijn_pip 3 links hand', 'Pijn_pip 3 rechts hand',
 'Pijn_pip 4 links hand', 'Pijn_pip 4 rechts hand', 'Pijn_pip 5 links hand', 'Pijn_pip 5 rechts hand', 
 'Pijn_mcp 1 links', 'Pijn_mcp 1 rechts', 'Pijn_mcp 2 links', 'Pijn_mcp 2 rechts', 'Pijn_mcp 3 links', 'Pijn_mcp 3 rechts',
 'Pijn_mcp 4 links', 'Pijn_mcp 4 rechts', 'Pijn_mcp 5 links', 'Pijn_mcp 5 rechts', 'Pijn_IP links', 'Pijn_IP rechts', 'Pijn_schouder L', 'Pijn_schouder R', 'Pijn_Elleboog L','Pijn_elleboog R',
 'Pijn_knie links', 'Pijn_knie rechts',
            'Pijn_sternoclaviculair L', 'Pijn_sternoclaviculair R', 'Pijn_acromioclaviaculair L',
'Pijn_acromioclaviaculair R', 'Pijn_pip 2 links voet', 'Pijn_pip 2 rechts voet',
'Pijn_pip 3 links voet', 'Pijn_pip 3 rechts voet',  'Pijn_pip 4 links voet', 
'Pijn_pip 4 rechts voet', 'Pijn_pip 5 links voet', 'Pijn_pip 5 rechts voet',
'Pijn_onderste spronggewricht links', 'Pijn_onderste spronggewricht rechts',
'Pijn_bovenste spronggewicht links', 'Pijn_bovenste spronggewricht rechts',
               ]
    l_DAS44_sjc = ['Zwelling_pols L', 'Zwelling_pols R', 'Zwelling_pip 2 links hand',  'Zwelling_pip 2 rechts hand',
     'Zwelling_pip 3 links hand',  'Zwelling_pip 3 rechts hand', 'Zwelling_pip 4 links hand', 'Zwelling_pip 4 rechts hand',  'Zwelling_pip 5 links hand',
     'Zwelling_pip 5 rechts hand',  'Zwelling_mcp 1 links', 'Zwelling_mcp 1 rechts', 'Zwelling_mcp 2 links', 'Zwelling_mcp 2 rechts', 'Zwelling_mcp 3 links', 'Zwelling_mcp 3 rechts',
     'Zwelling_mcp 4 links', 'Zwelling_mcp 4 rechts', 'Zwelling_mcp 5 links', 'Zwelling_mcp 5 rechts', 
     'Zwelling_knie links', 'Zwelling_knie rechts', 'Zwelling_schouder L', 'Zwelling_schouder R', 'Zwelling_Elleboog L',
      'Zwelling_elleboog R', 'Zwelling_IP links', 'Zwelling_IP rechts'
            'Zwelling_sternoclaviculair L', 'Zwelling_sternoclaviculair R', 'Zwelling_acromioclaviaculair L',
'Zwelling_acromioclaviaculair R', 'Zwelling_pip 2 links voet', 'Zwelling_pip 2 rechts voet',
'Zwelling_pip 3 links voet', 'Zwelling_pip 3 rechts voet',  'Zwelling_pip 4 links voet', 
'Zwelling_pip 4 rechts voet', 'Zwelling_pip 5 links voet', 'Zwelling_pip 5 rechts voet',
'Zwelling_onderste spronggewricht links', 'Zwelling_onderste spronggewricht rechts',
'Zwelling_bovenste spronggewicht links', 'Zwelling_bovenste spronggewricht rechts'
               ]   
    
    snapshots = list(df[((df['PATNR']==patnr)&(df['STELLING']!='BSE'))]['DATUM'].unique()) 
    # BSE is sometimes measured before Mannequin
    bse_df = df[((df['PATNR']==patnr)&(df['STELLING']=='BSE'))].copy()
    bse_df = bse_df.sort_values(by='DATUM')
    bse_df = bse_df.dropna(subset=['XANTWOORD'])
    
    # DAS28 is sometimes measured before Mannequin?
    das28_df = df[((df['PATNR']==patnr)&(df['STELLING']=='DAS 28'))].copy()
    das28_df = das28_df.sort_values(by='DATUM')
    das28_df = das28_df.dropna(subset=['XANTWOORD'])
    
    if first_date == None:
        first_date = pd.to_datetime(snapshots[0], format='%Y-%m-%d', errors='ignore')
    else :
        first_date = pd.to_datetime(first_date, format='%Y-%m-%d', errors='ignore')
    #print(len(snapshots))
    #print(df[((df['PATNR']==patnr) & (new_df['STELLING']=='BSE'))]['DATUM_A'].unique())
    #rint(snapshots)
    
    if verbose : print('Patnr %s' % (patnr))
        
    l_das = []
    d_das = {}
    d_missing = {}
    df_das = pd.DataFrame(columns=['patnr','date', 'days', 'DAS28(3)', 'SJC', 'TJC', 'ESR', 'SJC44', 'TJC44', 'DAS44'])
    
    for snap in snapshots:
        days = (pd.to_datetime(snap, format='%Y-%m-%d', errors='ignore') - first_date).days 
        snapshot_df = df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING'].isin(['Zwelling', 'Pijn'])))].copy()
        snapshot_df = snapshot_df.assign(XANTWOORD=snapshot_df['STELLING'] + "_" + snapshot_df['XANTWOORD'])
        #print('Length of items %s at %s: %s' % (str(patnr), str(snap), str(len(snapshot_df))))
        if verbose : print('Date %s = %s days' % (str(snap), str(days)))
        c_tjc = 0
        c_sjc = 0
        c_tjc44 = 0
        c_sjc44 = 0
        esr = 0
        
        das = 0
        das44 = 0
        index = len(df_das)
        
        
        # Try and except is faster than checking with if (especially since we are dealing with exceptional cases)
        
        try : 
            no_pain = df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='Totaal pijnlijke gewrichten'))]['XANTWOORD'].iloc[i]
            if no_pain == 0:
                no_pain = 'Y'
            else :
                no_pain = 'N'
        except:
            no_pain = 'N'
        try : 
            no_swollen = df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='Totaal gezwollen gewrichten'))]['XANTWOORD'].iloc[i]
            if no_swollen == 0:
                no_swollen = 'Y'
            else :
                no_swollen = 'N'
        except:
            no_swollen = 'N'
            
        try : 
            no_pain = df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='Geen pijnlijke gewrichten'))]['XANTWOORD'].iloc[i]
        except:
            no_pain = 'N'
        
        try :
            no_swollen = df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='Geen gezwollen gewrichten'))]['XANTWOORD'].iloc[i]
        except:
            no_swollen = 'N'
            
        try : 
            no_pain = df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='Pijnlijke gewrichten'))]['XANTWOORD'].iloc[i]
            if no_pain == 'geen.':
                no_pain = 'Y'
            else :
                no_pain = 'N'
        except:
            no_pain = 'N'
        
        try : 
            no_swollen = df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='Gezwollen gewrichten'))]['XANTWOORD'].iloc[i]
            if no_swollen == 'geen.':
                no_swollen = 'Y'
            else :
                no_swollen = 'N'
        except:
            no_swollen = 'N'
            
        
            
        # Sometimes you'll find that ESR is nan - in that case continue searching
        l1 = len(df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='BSE'))]['XANTWOORD'])
        for i in range(l1):
            if esr == 0 or esr == np.nan:
                esr = float(df[((df['PATNR']==patnr) & (df['DATUM'] == snap) & (df['STELLING']=='BSE'))]['XANTWOORD'].iloc[i])
            else :
                break
        
        def nearest_before(items, pivot):
            if len([i for i in items if i <= pivot]) > 0:
                return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))
            else :
                return pivot + pd.DateOffset(days=999) # incredibly large number to prevent observation is carried forward
        def nearest_after(items, pivot):
            if len([i for i in items if i >= pivot]) > 0:
                return min([i for i in items if i >= pivot], key=lambda x: abs(x - pivot)) 
            else :
                return pivot + pd.DateOffset(days=999) # incredibly large number to prevent observation is carried backward

        
        # If you still don't find any ESR - look in range
        if esr == 0 or esr == np.nan:
            d1 = pd.to_datetime(bse_df['DATUM'], format='%Y-%m-%d', errors='ignore')
            d2 = pd.to_datetime(snap, format='%Y-%m-%d', errors='ignore')
            #print(d1, d2)
            early = nearest_before(d1, d2) 
            late = nearest_after(d1, d2)
            #print((abs(late-d2).days), (abs(d2-early).days))
            #print(type(early), type(d2))
            #print(type(late), type(d2))
            if ((abs(late-d2).days) < (abs(d2-early).days)) and (abs(late-d2).days) < 3:
                esr = float(df[((df['PATNR']==patnr)&(df['STELLING']=='BSE')&(df['DATUM'].str.contains(str(late))))]['XANTWOORD'].iloc[0])
                if verbose : print("0 ->", esr, "Imputed from",  late, "which is", abs(late-d2).days, "days after date", d2)
            
            elif (abs(d2-early).days) < 14 and early != None: # less than two weeks
                #print(df[((df['PATNR']==patnr)&(df['STELLING']=='BSE')&(df['DATUM'].str.contains(early)))]['XANTWOORD'])
                esr = float(df[((df['PATNR']==patnr)&(df['STELLING']=='BSE')&(df['DATUM'].str.contains(str(early))))]['XANTWOORD'].iloc[0])
                if verbose : print("0 ->", esr, "Imputed from",  early, "which is", abs(d2-early).days, "days before date", d2)
        
        
            
        # Only calculate DAS if there is mannequin information
        # easy check by looking for joints. If there are no joints, look for 'Geen pijnlijke gewrichten'
        # and 'Geen gezwollen gewrichten'
        if esr != 0 and esr != np.nan:
            l_history= []
            l_history44 = []
            for key in snapshot_df['XANTWOORD']:
                if  key in l_DAS28_tjc and key not in l_history: 
                    c_tjc += 1
                    l_history.append(key)
                elif key in l_DAS28_sjc and key not in l_history: 
                    c_sjc += 1
                    l_history.append(key)
                
                if  key in l_DAS28_tjc and key not in l_history44: 
                    c_tjc44 += 1
                    l_history44.append(key)
                elif key in l_DAS44_sjc and key not in l_history44: 
                    c_sjc44 += 1
                    l_history44.append(key)
            #print(len(snapshot_df['XANTWOORD']), snapshot_df['XANTWOORD'], c_sjc, c_tjc)
        #print(no_pain, no_swollen)
        #if snap == '2015-08-11 00:00:00.000':
        #    print(str(patnr), str(snap), str(len(snapshot_df)), esr, c_tjc, c_sjc)
        if ((c_tjc + c_sjc) != 0 or (no_pain=='Y' and no_swollen=='Y')) and esr != 0:
            das = calculate_DAS28(c_tjc, c_sjc, esr)
            das44 = calculate_DAS44(c_tjc44, c_sjc44, esr)
            d_das[days] = das
            df_das.loc[index] = [patnr, snap, days, das, c_sjc, c_tjc, esr, c_sjc44, c_tjc44, das44]
            #l_das.append(das)
        else :
            # DAS28
            if len(df[((df['PATNR']==patnr)&(df['STELLING'].isin(['DAS 28', 'DAS 28(3)']))& (df['DATUM'].str.contains(str(snap))))]['XANTWOORD']) > 0: # look for DAS
                if verbose : print('Cannot compute DAS28; will extract directly from table', snap)
                das = df[((df['PATNR']==patnr)&(df['STELLING'].isin(['DAS 28', 'DAS 28(3)']))& (df['DATUM'].str.contains(str(snap))))]['XANTWOORD'].iloc[0]
                #print(das)
                if type(das) != float: # the raw value should be a string type
                    if 'VALUE1' in das:
                        das = re.search("VALUE1\=(\d+\,?(?:\d+)?)", das).group(1)
                    das = float(das.replace(",", ".")) # english notation
                    d_das[days] = das
                    df_das.loc[index] = [patnr, snap, days, das, c_sjc, c_tjc, esr, c_sjc44, c_tjc44, das44]
                #l_das.append(das)
            else :
                if esr == 0 or esr == np.nan:
                    if verbose : print('Cannot compute DAS28: ESR is missing at', snap)
                else :    
                    if verbose : print('Cannot compute DAS28: joint information is missing at', snap)
                d_missing[days] = 0
                
            # DAS44    
            if len(df[((df['PATNR']==patnr)&(df['STELLING'].isin(['DAS 44', 'DAS 44(3)']))& (df['DATUM'].str.contains(str(snap))))]['XANTWOORD']) > 0: # look for DAS
                if verbose : print('Cannot compute DAS44; will extract directly from table', snap)
                das44 = df[((df['PATNR']==patnr)&(df['STELLING'].isin(['DAS 44', 'DAS 44(3)']))& (df['DATUM'].str.contains(str(snap))))]['XANTWOORD'].iloc[0]
                #print(das)
                if type(das44) != float: # the raw value should be a string type
                    if 'VALUE1' in das44:
                        das44 = re.search("VALUE1\=(\d+\,?(?:\d+)?)", das44).group(1)
                    das44 = float(das44.replace(",", ".")) # english notation
                    df_das.loc[index] = [patnr, snap, days, das, c_sjc, c_tjc, esr, c_sjc44, c_tjc44, das44]
                #l_das.append(das)
            else :
                if esr == 0 or esr == np.nan:
                    if verbose : print('Cannot compute DAS44: ESR is missing at', snap)
                else :    
                    if verbose : print('Cannot compute DAS44: joint information is missing at', snap)
                d_missing[days] = 0
                
                
            # DAS 44(3)
            #except IndexError:
            #    continue
            
        #if snap == '2016-01-20 00:00:00.000':
        #    print(str(patnr), str(snap), str(len(snapshot_df)), esr, c_tjc, c_sjc)
        
    return d_das, d_missing, df_das

def calculate_DAS28(tjc, sjc, esr):
    """
    Calculate DAS28 with 3 variables : TJC, SJC and ESR (BSE)
    """
    das28 = (0.56 * np.sqrt(tjc) + 0.28 * np.sqrt(sjc) + 0.70 * np.log(esr)) * 1.08 + 0.16
    return das28

def calculate_DAS44(tjc, sjc, esr):
    """
    Calculate DAS44 with 3 variables : RAI, SJC and ESR (BSE)
    """
    
    das44= (0.53938 * np.sqrt(tjc) + 0.0650 * (sjc) + 0.330 * np.log(esr)) + 0.224 #0.224
    return das44


#d_bin = {}
#bin_size =3
l_pat = []
for pat in list(new_df['PATNR'].unique())[:250]:
    l_pat.append(pat)

def collect_all_das(df, bin_size=20, first_dates=None, verbose=False):
    """
    Collect all DAS values by computing the DAS or by extracting the DAS from the table.
    
    All of the DAS values are then exported to a standardized table
    """
    df_ultimate_das = pd.DataFrame(columns=['patnr','date', 'days', 'DAS28(3)', 'SJC', 'TJC', 'ESR']) 
    
    for pat in list(df['PATNR'].unique()):
        d_das, d_missing, df_das = compute_DAS(df, patnr=pat, first_date=first_dates[pat], verbose=verbose)
        df_ultimate_das = pd.concat([df_ultimate_das, df_das])
        
        print('patnr:', pat) # , d_das, first_dates[pat]
        mylist = np.array(list(d_das.keys()))

    return d_das, df_ultimate_das


BIN_SIZE = 20  
d_das, df_das = collect_all_das(new_df, BIN_SIZE, first_dates=new_pat, verbose=False)

# Export DAS information to csv
df_das.to_csv('data/8_final/DAS_patients.csv', index=False)