import re
import math
import numpy as np
import pandas as pd
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, CustomJS, Panel, DataTable, ColumnDataSource, Tabs, TableColumn
from bokeh.plotting import figure, show,  output_notebook
from bokeh.layouts import row, column, gridplot
from bokeh.transform import factor_cmap
from bokeh.models import  CategoricalColorMapper, LinearColorMapper, Select,  Slider
from bokeh.io import output_file, show
import bokeh.io as bio
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from numpy.linalg import norm
import umap
import pattern.nl as patNL
import pattern.de as patDE
import pattern.en as patEN

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
    
    
def reformat_ddrA(df, pat_col='PATNR', time_col='DATUM'): # patient_data, 
    """
    Cast DDR_A questionaire file from long to wide format. 
    Entries are merged on patient number.
    
    Input:
        df = dataframe (default HIX-output)
        pat_col = the patient identifier column
        time_col = the column with the date
    Output:
        o_df = output dataframe where each variable is its own column
    """
    print("\nProcessing lab data...") # PseudoID # patient_id'
    # Sort df - Keep the first measure -> cause we want to get the baseline characteristics
    df.sort_values([pat_col, time_col], ascending=False, ignore_index=True, inplace=True)
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
    for pat in df['patient_id'].unique():
        df_sub = df[((df['patient_id']==pat) & (df['test_naam_omschrijving'].isin(val)))]
        try :
            for i in range(len(df_sub)):
                result = df_sub['uitslag_text'].iloc[0]
                if result not in ['-volgt-', '@volgt'] : # stopgezet?
                    d_val[pat] = result
        except:
            d_val[pat] = np.nan
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
    return feature
        

def reformat_lab(df, pat_col='patient_id', time_col='Monster_Afname_Datumtijd'):
    """
    Cast raw LAB file from long to wide format. 
    Entries are merged on patient number (patient_id).
    
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
    o_df = pd.DataFrame(columns=['patnr', 'time'])
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
        
        
        feature = fuzzy_feature(descr, unit, value)
        

        # Make absolute time point relative to first symptoms day
        #time = abs_to_rel_date(patient_data['symptoms_date'].loc[pat_col], o_time)
        time = o_time

        # Add new column to output dataframe if feature has not yet been made
        if feature not in o_df.columns:
            o_df[feature] = np.nan

        # Populate row in output dataframe
        o_df.at[o_row, 'patnr'] = pseudo_id
        o_df.at[o_row, feature] = value
        o_df.at[o_row, 'time'] = time

    return o_df

def makeTSNE_Cluster2(values, l_id, l_val, l_lbl, title, pal, perp=30, legend=['RF', 'aCCP'], l_order=[], seed=1234):
    """
    
    Perform a t-SNE dimension reduction and render an interactive bokeh plot
    
    Input:
        values = datapoints on which PCA should be performed
        l_id = list of (patient) identifiers (patient id)
        l_val = list of second lable
        l_lbl = list of labels associated to datapoints (often categories)
        title = String containing title of plot
        lgend = list with labels of the two provided columns
        pal = pallete
        perp = perplexity for t-SNE
        seed = random seed
    """
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A T-SNE projection of %s patients" % (len(l_id)), tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)

    # dimensionality reduction. converting the vectors to 2d vectors
    tsne_model = TSNE(n_components=2, verbose=1, perplexity=perp, random_state=seed)
    tsne_2d = tsne_model.fit_transform(values)
    
    color_mapper = CategoricalColorMapper(factors=list(set(l_val)), palette=pal[0])
    
    if l_order != []:
        color_mapper2 = CategoricalColorMapper(factors=l_order, palette=pal[1])
    else :
        color_mapper2 = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal[1])
    
    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_2d, columns=['x', 'y'])
    tsne_df['pt'] = l_id
    tsne_df['label'] = l_lbl 
    tsne_df['value'] = l_val
    
    p1 = bp.figure(plot_width=700, plot_height=600, title="A T-SNE projection of %s patients" % (len(l_id)),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    
    p1.scatter(x='x', y='y', source=tsne_df, legend_field='label', size=10,  color={'field': 'label', 'transform': color_mapper2}) # fill_color=mapper
    
    hover = p1.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"pt": "@pt",legend[0] : "@value", legend[1]: "@label"}
    
    tab1 = Panel(child=p1, title=legend[1])
    
    p2 = bp.figure(plot_width=700, plot_height=600, title="A T-SNE projection of %s patients" % (len(l_id)),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_type=None, y_axis_type=None, min_border=1)
    p2.scatter(x='x', y='y', source=tsne_df, legend_field='value', size=10,  color={'field': 'value', 'transform': color_mapper})
    
    hover2 = p2.select(dict(type=HoverTool)) # or p1
    hover2.tooltips={"pt": "@pt", legend[0] : "@value", legend[1]: "@label"}
    
    tab2 = Panel(child=p2, title=legend[0])
    
    tabs = Tabs(tabs=[  tab2, tab1])

    bp.output_file('../TSNE/Baseline_tsne_%s.html' % (title), mode='inline')

    bp.save(tabs)
    print('\nTSNE figure saved under location: TSNE/Baseline_tsne_%s.html' % (title))
    return 

def makePCA(values, l_id, l_lbl, title, pal, radius=0.05, l_order=[], seed=1234):
    """
    Perform Principal Component Analysis for dimension reduction
    
    values = datapoints on which PCA should be performed
    l_id = list of (patient) identifiers associated to datapoints 
    l_lbl = list of labels associated to datapoints
    title = String containing title of plot
    pal = pallete
    radius = radius of the points to draw 
    l_order = list to indicate how labels should be sorted for coloring
    seed = random seed
    """
    # dimensionality reduction. converting the vectors to 2d vectors
    pca_model = PCA(n_components=2, random_state=seed) # , verbose=1, random_state=0
    pca_2d = pca_model.fit_transform(values)
    print('Explained PCA:\tPC1=', pca_model.explained_variance_ratio_[0], '\tPC2=',pca_model.explained_variance_ratio_[1])
    if l_order != []:
        color_mapper = CategoricalColorMapper(factors=l_order, palette=pal)
    else :
        color_mapper = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal)

    # putting everything in a dataframe
    pca_df = pd.DataFrame(pca_2d, columns=['x', 'y'])
    pca_df['pt'] = l_id
    pca_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A PCA projection of %s patients (Explained PCA:\tPC1=%.2f\tPC2=%.2f)" %(len(l_id), pca_model.explained_variance_ratio_[0], pca_model.explained_variance_ratio_[1]),        tools="pan,wheel_zoom,box_zoom,reset,hover,save", x_axis_type=None, y_axis_type=None, min_border=1)
    
    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=pca_df, legend_field="label", radius=radius,  color={'field': 'label', 'transform': color_mapper}) # fill_color=mapper
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"pt": "@pt"}
    bp.output_file('../TSNE/Baseline_pca_%s.html' % (title), mode='inline')
    bp.save(plot_tfidf)
    print('PCA figure saved under location: TSNE/Baseline_pca_%s.html' % (title))
    return

def makeMCA(values, l_id, l_lbl, title, pal, radius=0.05, l_order=[], seed=1234):
    """
    Perform MCA for dimension reduction
    
    values = datapoints on which PCA should be performed
    l_id = list of (patient) identifiers associated to datapoints 
    l_lbl = list of labels associated to datapoints
    title = String containing title of plot
    pal = pallete
    radius = radius of the points to draw 
    l_order = list to indicate how labels should be sorted for coloring
    seed = random seed
    """
    # dimensionality reduction. converting the vectors to 2d vectors
    
    mca_model = FactorAnalysis(n_components=2, random_state=0)
    mca_2d = mca_model.fit_transform(values)
    print('Explained PCA:\tPC1=', 15, '\tPC2=',10)
    if l_order != []:
        color_mapper = CategoricalColorMapper(factors=l_order, palette=pal)
    else :
        color_mapper = CategoricalColorMapper(factors=list(set(l_lbl)), palette=pal)

    # putting everything in a dataframe
    mca_df = pd.DataFrame(mca_2d, columns=['x', 'y'])
    print(len(mca_df), len(l_id), len(l_lbl))
    print(len(l_lbl.unique()))
    mca_df['pt'] = l_id
    mca_df['label'] = l_lbl #df.cc.astype('category').cat.codes
    
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A MCA projection of %s patients)" %(len(l_id)),        tools="pan,wheel_zoom,box_zoom,reset,hover,save", x_axis_type=None, y_axis_type=None, min_border=1)
    
    # plotting. the corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=mca_df, legend_field="label", radius=radius,  color={'field': 'label', 'transform': color_mapper}) # fill_color=mapper
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"pt": "@pt"}
    bp.output_file('../TSNE/Baseline_mca_%s.html' % (title), mode='inline')
    bp.save(plot_tfidf)
    print('MCA figure saved under location: TSNE/Baseline_mca_%s.html' % (title))
    return

def elbowMethod(X_trans, method='kmeans', n=20, save_as=''):
    """
    Define optimal number of clusters with elbow method, optimized for 
    Within cluster sum of errors(wcss).
    
    Input:
        X_trans = Distance matrix based on HPO binary data (X)
        method = clustering method
        n = search space (number of clusters to consider)
        save_as = title used for saving the elbow plot figure
            (no title implies that the figure won't be saved)
    
    Return:
        k = Number of clusters that corresponds to optimized WCSS
    """
    methods = {'kmeans' : KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0), 
              'fuzz' : []}
    lbl = {'kmeans' : ['Number of clusters', 'WCCS'], 
          'fuzz' : ['Number of clusters', 'fuzzy partition coefficient']}
    
    wcss = []
    distances = []
    fig1, ax1 = plt.subplots(1,2,figsize=(12,6))
    
    kmeans = methods[method]
    
    for i in range(1, n+1):
        if method == 'kmeans':
            kmeans.n_clusters = i
            kmeans.fit(X_trans)
            wcss.append(kmeans.inertia_)
        elif method == 'fuzz':
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_trans, i, 2, error=0.005, maxiter=1000, init=123)
            wcss.append(fpc)

    for i in range(1, len(wcss)+1):
        p1=np.array((1,wcss[0]))
        p2=np.array((n,wcss[len(wcss)-1]))
        p3 =np.array((i+1, wcss[i-1]))
        distances.append(norm(np.cross(p2-p1, p1-p3))/norm(p2-p1))
        
    k = distances.index(max(distances))+1

    ax1[0].plot(range(1, len(wcss)+1), wcss)
    ax1[0].set_title('Elbow Method')
    ax1[0].set_xlabel(lbl[method][0])
    ax1[0].set_ylabel(lbl[method][1])
    
    ax1[1].plot(range(1, len(wcss)+1), distances, color='r')
    ax1[1].plot([k, k], [max(distances),0],
                 color='navy', linestyle='--')
    ax1[1].set_title('Perpendicular distance to line between first and last observation')
    ax1[1].set_xlabel('Number of clusters')
    ax1[1].set_ylabel('Distance')
    
    if save_as != '':
        plt.savefig('../figures/3_clustering/elbow_plot_%s' % (save_as))
    else : 
        plt.show()
    return k

def make_tsne(dist, l_cat, identifier,perp=30, seed=1234, df=None, title=''):
    """
    Generate tsne
    
    dist= dataframe with distances
    df = dataframe with columns (in case dist only features the distances)
    """
    if type(df) == float:
        df = dist.copy()
    values = dist
    first_col = df[l_cat[0]] # c_first
    l_hex = ['#f00', '#fe0500', '#fc0a00', '#fb0f00', '#fa1400', '#f91900', '#f71e00', '#f62200', '#f52700', '#f42c00', '#f23000', '#f13500', '#f03a00', '#ee3e00', '#ed4200', '#ec4700', '#eb4b00', '#e94f00', '#e85400', '#e75800', '#e65c00', '#e46000', '#e36400', '#e26800', '#e16c00', '#df7000', '#de7300', '#d70', '#db7b00', '#da7f00', '#d98200', '#d88600', '#d68900', '#d58d00', '#d49000', '#d39300', '#d19700', '#d09a00', '#cf9d00', '#cda000', '#cca300', '#cba600', '#caa900', '#c8ac00', '#c7af00', '#c6b200', '#c5b500', '#c3b800', '#c2ba00', '#c1bd00', '#bfbf00', '#babe00', '#b5bd00', '#b0bc00', '#acba00', '#a7b900', '#a2b800', '#9db700', '#98b500', '#94b400', '#8fb300', '#8ab200', '#86b000', '#81af00', '#7dae00', '#79ac00', '#74ab00', '#70aa00', '#6ca900', '#68a700', '#64a600', '#60a500', '#5ca400', '#58a200', '#54a100', '#50a000', '#4c9e00', '#489d00', '#459c00', '#419b00', '#3d9900', '#3a9800', '#369700', '#339600', '#2f9400', '#2c9300', '#299200', '#269100', '#228f00', '#1f8e00', '#1c8d00', '#198b00', '#168a00', '#138900', '#108800', '#0d8600', '#0b8500', '#088400', '#058300', '#038100', '#008000']
    c_first = [l_hex[round(i/max(df[l_cat[0]]) * 100)] for i in df[l_cat[0]]]
    
    # Run T-SNE
    tsne_model = TSNE(n_components=2, verbose=1, perplexity=perp, random_state=seed)
    tsne_2d = tsne_model.fit_transform(values)
    
    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_2d, columns=['x', 'y'])
    
    d_col = dict(x=tsne_df['x'], y=tsne_df['y'], pt=df[identifier], c_col=c_first)
    
    for cat in l_cat:
        d_col[cat] = df[cat]
    
    s1 = ColumnDataSource(data=d_col)
    
    tsne_df['pt'] = df[identifier]
    tsne_df['value'] = first_col
    
    p3 = figure(plot_width=700, plot_height=600, tools="pan,wheel_zoom,box_zoom,reset,hover,save", title="A T-SNE projection of %s patients" % (len(first_col)))
    cir = p3.circle('x', 'y', source=s1, alpha=1, line_color='c_col',  fill_color='c_col')

    color_select = Select(title="color", value=l_cat[0], 
                        options = l_cat,)
    color_select.js_on_change('value', CustomJS(args=dict(cir=cir,s1=s1),
                                          code="""                                                             
        var data = s1.data;
        var gradient = ['#f00', '#fe0500', '#fc0a00', '#fb0f00', '#fa1400', '#f91900', '#f71e00', '#f62200', '#f52700', '#f42c00', '#f23000', '#f13500', '#f03a00', '#ee3e00', '#ed4200', '#ec4700', '#eb4b00', '#e94f00', '#e85400', '#e75800', '#e65c00', '#e46000', '#e36400', '#e26800', '#e16c00', '#df7000', '#de7300', '#d70', '#db7b00', '#da7f00', '#d98200', '#d88600', '#d68900', '#d58d00', '#d49000', '#d39300', '#d19700', '#d09a00', '#cf9d00', '#cda000', '#cca300', '#cba600', '#caa900', '#c8ac00', '#c7af00', '#c6b200', '#c5b500', '#c3b800', '#c2ba00', '#c1bd00', '#bfbf00', '#babe00', '#b5bd00', '#b0bc00', '#acba00', '#a7b900', '#a2b800', '#9db700', '#98b500', '#94b400', '#8fb300', '#8ab200', '#86b000', '#81af00', '#7dae00', '#79ac00', '#74ab00', '#70aa00', '#6ca900', '#68a700', '#64a600', '#60a500', '#5ca400', '#58a200', '#54a100', '#50a000', '#4c9e00', '#489d00', '#459c00', '#419b00', '#3d9900', '#3a9800', '#369700', '#339600', '#2f9400', '#2c9300', '#299200', '#269100', '#228f00', '#1f8e00', '#1c8d00', '#198b00', '#168a00', '#138900', '#108800', '#0d8600', '#0b8500', '#088400', '#058300', '#038100', '#008000'];    

        var selected_color = cb_obj.value;

        console.log(cb_obj.value)
        
        data["desc"] = [] ;
        for (var i=0;i<data["x"].length; i++) {
        data["desc"].push(data[selected_color][i]);
        };

        var max = data[selected_color].reduce(function(a, b) {
            return Math.max(a, b);
        });

        data["c_col"] = [] ;
        for (var i=0;i<data["x"].length; i++) {
        var ix = (Math.floor((data[selected_color][i]/max) * 100))
        data["c_col"].push(gradient[ix]);
        };

        

        cir.glyph.line_color.field = "c_col";
        cir.glyph.fill_color.field = "c_col";

        s1.change.emit()

    """)) # dict[cb_obj.value]

    hover = p3.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"ID": "@pt","value" : "@desc"}

    layout = gridplot([[p3],[color_select]]) # 

    bio.output_file("../TSNE/Baseline_tsne_%s.html" % (title), mode='inline')
    bio.show(layout)

    print('\nTSNE figure saved under location: TSNE/Baseline_tsne_%s.html' % (title))
    return

def make_umap(embedding, df, l_cat, identifier,perp=30, seed=1234, title=''):
    """
    Make UMAP
    
    embedding= dataframe / 2d array with distances
    df = dataframe with columns (in case dist only features the distances)
    """
    #if type(df) == float:
    #    df = embedding.copy()
    #values = embedding
    first_col = df[l_cat[0]] # c_first
    l_hex = ['#f00', '#fe0500', '#fc0a00', '#fb0f00', '#fa1400', '#f91900', '#f71e00', '#f62200', '#f52700', '#f42c00', '#f23000', '#f13500', '#f03a00', '#ee3e00', '#ed4200', '#ec4700', '#eb4b00', '#e94f00', '#e85400', '#e75800', '#e65c00', '#e46000', '#e36400', '#e26800', '#e16c00', '#df7000', '#de7300', '#d70', '#db7b00', '#da7f00', '#d98200', '#d88600', '#d68900', '#d58d00', '#d49000', '#d39300', '#d19700', '#d09a00', '#cf9d00', '#cda000', '#cca300', '#cba600', '#caa900', '#c8ac00', '#c7af00', '#c6b200', '#c5b500', '#c3b800', '#c2ba00', '#c1bd00', '#bfbf00', '#babe00', '#b5bd00', '#b0bc00', '#acba00', '#a7b900', '#a2b800', '#9db700', '#98b500', '#94b400', '#8fb300', '#8ab200', '#86b000', '#81af00', '#7dae00', '#79ac00', '#74ab00', '#70aa00', '#6ca900', '#68a700', '#64a600', '#60a500', '#5ca400', '#58a200', '#54a100', '#50a000', '#4c9e00', '#489d00', '#459c00', '#419b00', '#3d9900', '#3a9800', '#369700', '#339600', '#2f9400', '#2c9300', '#299200', '#269100', '#228f00', '#1f8e00', '#1c8d00', '#198b00', '#168a00', '#138900', '#108800', '#0d8600', '#0b8500', '#088400', '#058300', '#038100', '#008000']
    c_first = [l_hex[round(i/max(df[l_cat[0]]) * 100)] for i in df[l_cat[0]]]
    
    # Run UMAP
    #mapper_2d = umap.UMAP().fit_transform(values)
    
    # putting everything in a dataframe
    umap_df = pd.DataFrame(embedding, columns=['x', 'y'])
    
    d_col = dict(x=umap_df['x'], y=umap_df['y'], pt=df[identifier], c_col=c_first)
    
    for cat in l_cat:
        d_col[cat] = df[cat]
    
    s1 = ColumnDataSource(data=d_col)
    
    umap_df['pt'] = df[identifier]
    umap_df['value'] = first_col
    
    p3 = figure(plot_width=700, plot_height=600, tools="pan,wheel_zoom,box_zoom,reset,hover,save", title="A T-SNE projection of %s patients" % (len(first_col)))
    cir = p3.circle('x', 'y', source=s1, alpha=1, line_color='c_col',  fill_color='c_col')

    color_select = Select(title="color", value=l_cat[0], 
                        options = l_cat,)
    color_select.js_on_change('value', CustomJS(args=dict(cir=cir,s1=s1),
                                          code="""                                                             
        var data = s1.data;
        var gradient = ['#f00', '#fe0500', '#fc0a00', '#fb0f00', '#fa1400', '#f91900', '#f71e00', '#f62200', '#f52700', '#f42c00', '#f23000', '#f13500', '#f03a00', '#ee3e00', '#ed4200', '#ec4700', '#eb4b00', '#e94f00', '#e85400', '#e75800', '#e65c00', '#e46000', '#e36400', '#e26800', '#e16c00', '#df7000', '#de7300', '#d70', '#db7b00', '#da7f00', '#d98200', '#d88600', '#d68900', '#d58d00', '#d49000', '#d39300', '#d19700', '#d09a00', '#cf9d00', '#cda000', '#cca300', '#cba600', '#caa900', '#c8ac00', '#c7af00', '#c6b200', '#c5b500', '#c3b800', '#c2ba00', '#c1bd00', '#bfbf00', '#babe00', '#b5bd00', '#b0bc00', '#acba00', '#a7b900', '#a2b800', '#9db700', '#98b500', '#94b400', '#8fb300', '#8ab200', '#86b000', '#81af00', '#7dae00', '#79ac00', '#74ab00', '#70aa00', '#6ca900', '#68a700', '#64a600', '#60a500', '#5ca400', '#58a200', '#54a100', '#50a000', '#4c9e00', '#489d00', '#459c00', '#419b00', '#3d9900', '#3a9800', '#369700', '#339600', '#2f9400', '#2c9300', '#299200', '#269100', '#228f00', '#1f8e00', '#1c8d00', '#198b00', '#168a00', '#138900', '#108800', '#0d8600', '#0b8500', '#088400', '#058300', '#038100', '#008000'];    

        var selected_color = cb_obj.value;

        console.log(cb_obj.value)
        
        data["desc"] = [] ;
        for (var i=0;i<data["x"].length; i++) {
        data["desc"].push(data[selected_color][i]);
        };

        var max = data[selected_color].reduce(function(a, b) {
            return Math.max(a, b);
        });

        data["c_col"] = [] ;
        for (var i=0;i<data["x"].length; i++) {
        var ix = (Math.floor((data[selected_color][i]/max) * 100))
        data["c_col"].push(gradient[ix]);
        };

        

        cir.glyph.line_color.field = "c_col";
        cir.glyph.fill_color.field = "c_col";

        s1.change.emit()

    """)) # dict[cb_obj.value]

    hover = p3.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"ID": "@pt","value" : "@desc"}

    layout = gridplot([[p3],[color_select]]) # 
    

    bio.output_file("../TSNE/Baseline_umap_%s.html" % (title), mode='inline')
    bio.show(layout)

    print('\nUMAP figure saved under location: TSNE/Baseline_umap_%s.html' % (title))
    return

def visualize_umap_bokeh(embedding, df, l_cat, l_binary=[], patient_id='patnr', cluster_id='PhenoGraph_clusters', title='', path=None):
    """
    This function generates a bokeh scatter plot based on the provided embedding 
    (which is generated by a dimension reduction technique)
    
    Input:
        embedding= dataframe / 2d array with distances
        df = dataframe with columns (in case dist only features the distances)
        l_cat = specify columns to showcase
        l_binary = indicates the binary columns where the prevalence should be calculated instead of the mean!
        patient_id = str indicating column of patient
        cluster_id = str indicating column of clusters
        title = title of the bokeh plot
        path = str indicating the path where to save the file
    
    Output:
        Interactive HTML with a UMAP render
    """
    cluster_ix = 0

    first_col = df[l_cat[0]] # c_first
    l_hex = ['#f00', '#fe0500', '#fc0a00', '#fb0f00', '#fa1400', '#f91900', '#f71e00', '#f62200', '#f52700', '#f42c00', '#f23000', '#f13500', '#f03a00', '#ee3e00', '#ed4200', '#ec4700', '#eb4b00', '#e94f00', '#e85400', '#e75800', '#e65c00', '#e46000', '#e36400', '#e26800', '#e16c00', '#df7000', '#de7300', '#d70', '#db7b00', '#da7f00', '#d98200', '#d88600', '#d68900', '#d58d00', '#d49000', '#d39300', '#d19700', '#d09a00', '#cf9d00', '#cda000', '#cca300', '#cba600', '#caa900', '#c8ac00', '#c7af00', '#c6b200', '#c5b500', '#c3b800', '#c2ba00', '#c1bd00', '#bfbf00', '#babe00', '#b5bd00', '#b0bc00', '#acba00', '#a7b900', '#a2b800', '#9db700', '#98b500', '#94b400', '#8fb300', '#8ab200', '#86b000', '#81af00', '#7dae00', '#79ac00', '#74ab00', '#70aa00', '#6ca900', '#68a700', '#64a600', '#60a500', '#5ca400', '#58a200', '#54a100', '#50a000', '#4c9e00', '#489d00', '#459c00', '#419b00', '#3d9900', '#3a9800', '#369700', '#339600', '#2f9400', '#2c9300', '#299200', '#269100', '#228f00', '#1f8e00', '#1c8d00', '#198b00', '#168a00', '#138900', '#108800', '#0d8600', '#0b8500', '#088400', '#058300', '#038100', '#008000']
    
    
    c_first = [l_hex[round(i/max(df[l_cat[0]]) * 100)] for i in df[l_cat[0]]]

    c_alpha = [1 if i == cluster_ix else 0.1 for i in df[cluster_id] ]
    
    # putting everything in a dataframe
    umap_df = pd.DataFrame(embedding, columns=['x', 'y'])
    
    d_col = dict(x=umap_df['x'], y=umap_df['y'], pt=df[patient_id], c_col=c_first, c_alpha=c_alpha)
    
    # Add cluster column in case user didn't specify it in l_cat
    if cluster_id not in l_cat:
        l_cat.append(cluster_id)
    
    for cat in l_cat:
        d_col[cat] = df[cat]
    
    s1 = ColumnDataSource(data=d_col)
    
    umap_df['pt'] = df[patient_id]
    umap_df['value'] = first_col
    
    p3 = figure(plot_width=600, plot_height=500, tools="pan,wheel_zoom,box_zoom,reset,hover,save", title="An UMAP projection of %s patients" % (len(first_col)))
    cir = p3.circle('x', 'y', source=s1, alpha='c_alpha', line_color='c_col',  fill_color='c_col')

    color_select = Select(title="color", value=l_cat[0], 
                        options = l_cat,)
    color_select.js_on_change('value', CustomJS(args=dict(cir=cir,s1=s1),
                                          code="""                                                             
        var data = s1.data;
        var gradient = ['#f00', '#fe0500', '#fc0a00', '#fb0f00', '#fa1400', '#f91900', '#f71e00', '#f62200', '#f52700', '#f42c00', '#f23000', '#f13500', '#f03a00', '#ee3e00', '#ed4200', '#ec4700', '#eb4b00', '#e94f00', '#e85400', '#e75800', '#e65c00', '#e46000', '#e36400', '#e26800', '#e16c00', '#df7000', '#de7300', '#d70', '#db7b00', '#da7f00', '#d98200', '#d88600', '#d68900', '#d58d00', '#d49000', '#d39300', '#d19700', '#d09a00', '#cf9d00', '#cda000', '#cca300', '#cba600', '#caa900', '#c8ac00', '#c7af00', '#c6b200', '#c5b500', '#c3b800', '#c2ba00', '#c1bd00', '#bfbf00', '#babe00', '#b5bd00', '#b0bc00', '#acba00', '#a7b900', '#a2b800', '#9db700', '#98b500', '#94b400', '#8fb300', '#8ab200', '#86b000', '#81af00', '#7dae00', '#79ac00', '#74ab00', '#70aa00', '#6ca900', '#68a700', '#64a600', '#60a500', '#5ca400', '#58a200', '#54a100', '#50a000', '#4c9e00', '#489d00', '#459c00', '#419b00', '#3d9900', '#3a9800', '#369700', '#339600', '#2f9400', '#2c9300', '#299200', '#269100', '#228f00', '#1f8e00', '#1c8d00', '#198b00', '#168a00', '#138900', '#108800', '#0d8600', '#0b8500', '#088400', '#058300', '#038100', '#008000'];    

        var selected_color = cb_obj.value;

        console.log(cb_obj.value)
        
        data["desc"] = [] ;
        for (var i=0;i<data["x"].length; i++) {
        data["desc"].push(data[selected_color][i]);
        };

        var max = data[selected_color].reduce(function(a, b) {
            return Math.max(a, b);
        });

        data["c_col"] = [] ;
        for (var i=0;i<data["x"].length; i++) {
        var ix = (Math.floor((data[selected_color][i]/max) * 100))
        data["c_col"].push(gradient[ix]);
        };

        

        cir.glyph.line_color.field = "c_col";
        cir.glyph.fill_color.field = "c_col";

        s1.change.emit()

    """)) # dict[cb_obj.value]

    hover = p3.select(dict(type=HoverTool)) # or p1
    hover.tooltips={"ID": "@pt","value" : "@desc"}
    
    
    def getSummary(col, cluster, cluster_ix):
        if col.name == cluster_id:
            return cluster_ix
        elif col.dtype == float:
            return col.mean()
        elif col.dtype == int and col.max() < 3:
            return len(col[col==1])/len(col) #col.value_counts()#/len(col)
        elif col.dtype == int and col.max() > 2:
            return col.mean()
    
    
    new_df = pd.DataFrame() 
    # by default cluster 0
    new_df[0] = df[df[cluster_id]==cluster_ix][l_cat].apply(lambda x : getSummary(x, cluster_id, cluster_ix))
    new_df = new_df.reset_index()
    new_df.columns = ['var', 'meanprev']
    
    s2 = ColumnDataSource(new_df)

    columns = [
            TableColumn(field="var", title="Variable"),
            TableColumn(field="meanprev", title="Mean or Prevalence"),
        ]
    tb = DataTable(source=s2 , columns=columns, width=400, height=280)


    alp = Slider(start=0, end=1, value=0.1, step=.01, title="Alpha")
    
    
    cluster_select = Select(title="Select cluster", value=str(cluster_ix), 
                        options = [str(i) for i in df[cluster_id].unique()])
    
    # console.log(cb_obj.value)
    # var l_lab = ['MCV', 'Leuko', 'MCH', 'Hb', 'Ht', 'MCHC', 'BSE', 'Trom', 'prediction', 'RF',  'aCCP', 'aSSA', 'ENA', 'ANA'];
    cluster_select.js_on_change('value', CustomJS(args=dict(tb=tb, s2=s2, s1=s1, alp=alp, l_lab=l_cat, l_binary=l_binary, clust = cluster_id),
                                          code="""
        var l_lab = l_lab;
        var l_cat = l_binary;
        var clust = clust;
        var data = s2.data;
        var all = s1.data;
        var alpha = alp.value;
        var selected_number = cb_obj.value;

        console.log(l_lab)
        
        data["meanprev"] = [] ;
        for (var j=0;j<l_lab.length; j++){
            var cat = l_lab[j];
            
            var l_rf = [];
            var sum = 0;
            
            for (var i=0;i<all['x'].length; i++) {
                if (all[clust][i] == cb_obj.value ) {
                    l_rf.push(all[cat][i]);
                    if (l_cat.includes(cat)) {
                        // if categorical => count prevalence
                        if (all[cat][i] == 1){
                            sum += 1;
                            };
                        } else {
                            // if numerical => calculate mean
                            sum += all[cat][i];
                        };
                    };
                };
            
            data["meanprev"].push(sum/l_rf.length);
        };
        data["meanprev"].push(selected_number);
        
        // change alpha ? 
        all["c_alpha"] = [];
        for (var i=0;i<all["x"].length; i++) {
            if (all[clust][i] == cb_obj.value ) {
                all["c_alpha"].push(1);
            } else {
                all["c_alpha"].push(alpha);
            };
        };
        
        
        s1.change.emit()
        s2.change.emit()
    """)) 

    alp.js_on_change('value', CustomJS(args=dict(tb=tb, s1=s1, cs=cluster_select, clust = cluster_id),
                                          code="""
        var all = s1.data;
        var alpha = cb_obj.value;
        var clust = clust;
        
        console.log(cb_obj.value)
        
        // change alpha ? 
        all["c_alpha"] = [];
        for (var i=0;i<all["x"].length; i++) {
            if (all[clust][i] == cs.value ) {
                all["c_alpha"].push(1);
            } else {
                all["c_alpha"].push(alpha);
            };
        };
        
        s1.change.emit()
                                          
                                          """))
    
    layout = gridplot([[p3, column(tb, alp, cluster_select)],[color_select, ]]) # 
    
    if path == None:
        bio.output_file("../TSNE/Baseline_umap_%s.html" % (title), mode='inline')
    else :
        bio.output_file(path, mode='inline')
    bio.show(layout)

    print('\nUMAP figure saved under location: TSNE/Baseline_umap_%s.html' % (title))
    return