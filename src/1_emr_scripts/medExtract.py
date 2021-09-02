"""
Extract medication information

Disclaimer : this is the old version (2019)
"""
PATH_FILES= r'../'
import sys
sys.path.append(r'/exports/reum/tdmaarseveen/modules/')
#import pyConTextNLP2.pyConTextNLP.pyConText as pyConText
#import pyConTextNLP2.pyConTextNLP.itemData as itemData
#from textblob import TextBlob
import re
#import networkx as nx
import pandas as pd
import io
PATH_FILES= r'../../' # root 

def list_to_df(conv_list):
    headers = conv_list.pop(0)
    df = pd.DataFrame(conv_list, columns=headers)
    return df

def myreadlines(f, newline):
    buf = ""
    while True:
        while newline in buf:
            pos = buf.index(newline)
            yield buf[:pos]
            buf = buf[pos + len(newline):]
        chunk = f.read(4096)
        if not chunk:
            yield buf
            break
        buf += chunk

def build_convenient_list(f_name, delim, encod, max_count=None, col=None, val=None, endchar='\n', verbose=False):
    """
    This ensures that you can read at least parse 
    some of the data 
    
    Input:
        f_name = name of file
        delim = seperator / delimiter (e.g= comma's or spaces)
        encod = encoding used
        col = column to select on
        val = desired value
        endchar = character that splits the rows (e.g. \n or [report_end]) 
        verbose= print extra information
    """
    
    if max_count == None:
        max_count = 99999999999999 # temporary fix
    tot_list = []
    content = ""
    count = 0
    
    with open(f_name, mode='r', encoding= encod, errors='ignore') as f:
        for line in myreadlines(f, endchar):
            if line == '':
                print('End of file reached')
                break
            # Get next line from file
            #print(line)
            #line = f.readline()
            line = line.replace('\x00', '')
            #print(line)
            if count == 0:
                if col != None: # assess position of column
                    try:
                        col_nr= line.split(col)[0].count(delim)
                        print('Column ' + str(col) + ' has colnr: ' + str(col_nr))
                        #content += line + endchar
                    except :
                        print('Column ' + str(col) + ' does not exist')
                        col = None
                nr_delim = line.count(delim)
                print('Nr of columns:', nr_delim)
                #print(len(line.split(delim)))
                new_df = pd.DataFrame(columns=line.split(delim))
            while line.count(delim) < nr_delim:
                l = f.readline()
                l = l.replace('\x00', '')
                line += l # concatenate till it is correct
            if line.count(delim) > nr_delim:
                print('ERROR more or less columns found than needed!!')

            if col == None:
                #content += line + endchar
                new_df.loc[len(new_df)] = line.split(delim)
            elif delim in line:
                try: 
                    if type(val) == list:
                        if line.split(delim)[col_nr] in val: # only select if value in list!
                            #content += line + endchar
                            new_df.loc[len(new_df)] = line.split(delim)
                            #new_df = new_df.append(line.split(delim), ignore_index=True)
                    else :
                        if line.split(delim)[col_nr] == str(val): # only select if matching value
                            #content += line + endchar
                            new_df.loc[len(new_df)] = line.split(delim)
                except :
                    print(line) # temporary fix! -> these lines are capped off early (undesirable -> should probably make new Hix extraction)
            count += 1
            if verbose :
                if count % 50000 == 0:
                    print(str(count)+ 'th iteration')
            # if line is empty
            # end of file is reached
            if count>max_count:
                break
    
    #for x in content.split(endchar): # or [report_end]
    #    if delim in x:
    #        tot_list.append(x.split(delim))
    return new_df #= new_df.append(sub_df, ignore_index=True)

def build_convenient_list_old(f_name, delim, encod, max_count=None, col=None, val=None, endchar='\n', verbose=False):
    """
    This ensures that you can read at least parse 
    some of the data 
    
    Input:
        f_name = name of file
        delim = seperator / delimiter (e.g= comma's or spaces)
        encod = encoding used
        col = column to select on
        val = desired value
        endchar = character that splits the rows (e.g. \n or [report_end]) 
        verbose = print extra information
    """
    
    if max_count == None:
        max_count = 99999999999999 # temporary fix
    tot_list = []
    if encod == "ascii":
        f = io.open(f_name, mode='r', encoding= encod, newline=endchar)
    else :
        f = io.open(f_name, mode='r', encoding= encod, errors='ignore', newline=endchar)
    content = ""
    count = 0

            
    while True:
        # Get next line from file
        
        line = f.readline()
        line = line.replace('\x00', '')
        line = line.replace('\n', '') # line.rstrip()
        line = line.replace('\r', '') # line.rstrip()
        
        if line == '':
            print('End of File')
            break
        
        #print(line)
        if count == 0:
            if col != None: # assess position of column
                try:
                    col_nr= line.split(col)[0].count(delim)
                    print('Column ' + str(col) + ' has colnr: ' + str(col_nr))
                    content += line
                except :
                    print('Column ' + str(col) + ' does not exist')
                    col = None
            nr_delim = line.count(delim)
            print('Nr of columns:', nr_delim)
            tot_list.append(line.split(delim))
        while line.count(delim) < nr_delim:
            l = f.readline()
            l = l.replace('\x00', '')
            l = l.replace('\n', '') # line.rstrip()
            l = l.replace('\r', '') # line.rstrip()
            line += l # concatenate till it is correct
        if line.count(delim) > nr_delim:
            print('ERROR more or less columns found than needed!!')
            continue

        if col == None:
            #content += line
            tot_list.append(line.split(delim))
        elif delim in line:
            try: 
                if type(val) == list:
                    if line.split(delim)[col_nr] in val: # only select if value in list!
                        #content += line #+ "[report_end]"
                        tot_list.append(line.split(delim))
                else :
                    if line.split(delim)[col_nr] == str(val): # only select if matching value
                        #content += line #+ "[report_end]"
                        tot_list.append(line.split(delim))
            except :
                print(line) # temporary fix! -> these lines are capped off early (undesirable -> should probably make new Hix extraction)
        count += 1
        if verbose :
            if count % 50000 == 0:
                print(str(count)+ 'th iteration')
                print(len(tot_list))
        # if line is empty
        # end of file is reached
        if not line or count>max_count:
            break
    count = 0
    print("Start splitting!")
    #for x in content.split(endchar): # or [report_end]
    #    if delim in x:
    #        tot_list.append(x.split(delim))
    #    count += 1
    #    if verbose :
    #        if count % 50000 == 0:
    #            print(str(count)+ 'th iteration when splitting')
    print(len(tot_list))
    f.close()
    return tot_list

class ContextProcessing(object):
    """
    Input: df (dataframe) -> context exploration or test set
    
    This class is tasked with processing the reports and extracting
    the relevant features. This class utilizes pyContext to achieve this.
    
    Variables:
        modifiers = list of possible modifiers a.k.a. 
            trigger terms that are related to the
            target. For example: strength of the drug (12.5 mg) or a 
            negation indicator (no or not or stop).
        targets = list of possible targets (provided 
            from a yml file) For example: Methotrexaat
    """
    
    def __init__(self, path_mod=PATH_FILES + r'corpus/modifiersNL.yml', \
                 path_tar=PATH_FILES + r'corpus/targets.yml'):
        self.modifiers = itemData.get_items(
            path_mod, url=False)
        self.targets = itemData.get_items(
            path_tar, url=False)
        d_dim = self.getDimensionalTriggers() 
        self.d_correct = {'Continue': d_dim['Continue'], 'Stop': d_dim['Stop'], \
                    'Start': d_dim['Start'], 'Afbouwen': d_dim['Afbouwen'], \
                    'Verhogen' : d_dim['Verhogen'], \
                    'Verlagen': d_dim['Verlagen'], 'Ophogen': d_dim['Ophogen'], \
                    'Switch' : d_dim['Verandering']}
    def getDF(self):
        return self.df
    
    def setDF(self, df):
        self.df = df
        return
    
    def getModifiers(self):
        return self.modifiers
    
    def getTargets(self):
        return self.targets
    
    def getDimensionalTriggers(self):
        """
        This function composes a dictionary of triggers and links
         them to the associated higher dimensional 
        category. 

        The triggers are at a later point utilized in regular 
        expressions to convert synonyms to a similar & standardized
        category.
        
        For example: 
            Staken, gestaakt and gestopt will all be translated 
            to the category Stop  
        """
        d = {}
        d['Continue'] = ['continue', 'door', 'volhouden', 'vervolg', 
         'doorzetten', 'iter', 'voortzetten', \
         'voortgezet', 'hervatten']
        d['Stop'] = ['stop', 'staken', 'staak']
        d['Start'] = ['toevoegen', 'start']
        d['Afbouwen'] = ['af te bouwen', 'afbouwen', 'afgebouwd']
        d['Verlagen'] = ['lagen', 'laagd', 'laag']
        d['Ophogen'] = ['op']
        d['Verhogen'] = ['verhogen', 'verhoogd', 'verhoog']
        d['Verandering'] = ['switch']
        return d
        
    def setModifiers(self, path_mod, path_tar):
        """
        The modifiers and targets are set accordingly 
        to the provided .yml file.

        Variables:
            path_mod = path to the modifier file
            path_tar = path to the target file 
        """ 
        self.modifiers = itemData.get_items(
            path_mod, url=False)
        self.targets = itemData.get_items(
            path_tar, url=False)
        return
    
    def markup_sentence(self, s, prune_inactive=True):
        """
        This function is concerned with marking the targets and
        modifiers as well as entity linking and relation extraction.
        Firstly, the object is called and the text of the report is 
        provided with setRawText(). This text is then cleaned with 
        cleanText(). Both the modifiers and targets are then marked

        Once the modifiers & targets are marked two pruning steps 
        are utilized:
            1. pruneMarks()
                modifiers that overlap with other modifiers
                are pruned. The modifiers that cover the largest
                portion of text are kept.
            2. pruneModifierRelationships()
                prunes modifiers that are linked to multiple 
                targets at once by only allowing the modifiers
                to be linked with the closest target.
            3. pruneSelfModifyingRelationships()
                modifiers that are also part of the target are 
                pruned. So a modifier inside a target is unable
                to modify itself. 
        Finally all of the inactive modifiers (modifiers without
        a target) are removed from the markup object.
        """
        markup = pyConText.ConTextMarkup()
        markup.setRawText(s)
        markup.cleanText()
        
        markup.markItems(self.modifiers, mode="modifier")
        markup.markItems(self.targets, mode="target")
        
        markup.pruneMarks()
        
        markup.dropMarks('Exclusion')
        # apply modifiers to any targets within the modifiers scope
        markup.applyModifiers()
        markup.pruneSelfModifyingRelationships() #
        markup.pruneModifierRelationships()
        if prune_inactive:
            markup.dropInactiveModifiers()
        return markup
    
    def checkNan(self, val):
        """
        Checks whether a value equals Nan

        Variables:
            val = value to be checked
        """
        if type(val) == float:
            return ''
        else :
            return val
    
    def calculateConfidence(self, df):
        """
        This function calculates the confidence for each medication
        mentioned in the report. The measurement is based on a 
        collection of features. By default a confidence of 0.15 is 
        necessary for the program to recognize the mentioned drug 
        as part of a prescription.

        The minimal requirements for a prescription are: 
            - presence of a drug (0.05) 
            - Either one of the following elements:
                - strength
                - indicator for continuation
                - indicator for change
        The confidence increases according to the number of 
        features found. If there is a negation term near the 
        drug than the confidence value will be negatively 
        influenced.
        """
        conf = (0.05+0.1 * (df['strength_nr'] != None or df['strength_unit'] != None or  
                            df['continue'] == True or df['change'] == True) \
                + 0.05 * (df['dosage_nr'] != None)) 
        for x in ['route', 'form', 'duration', 'frequency']:
            if x == 'frequency':
                if df['freq_unit'] != None or df['freq_nr'] != None:
                    conf *= 1.5
            elif x == 'duration':
                if df['duration_unit'] != None or df['duration_nr'] != None:
                    conf *= 1.3
            elif df[x] != None:
                conf *= 1.3
        if (df['negation'] != None and df['negation'] != False):
            conf *= -1
        return conf
    
    def generateConclusion(self, d_features, target):
        """
        This function generates a string conclusion based on the 
        available features. The features are only appended if 
        a value is found (so feature != None). If there are no
        features at all than the conclusion is as follows:
        'geen voorschrift' (no prescription). The same 
        conclusion is provided in the following cases as well:
            - Hypothetical context
            - Confidence* below 0.15

        If the continuation of the drug is mentioned in the report 
        then the conclusion 'no change' is made. (In Dutch: geen 
        verandering). 

        The final conclusion string is returned as output.

        Variables:
            d_features = dictionary consisting of all available 
                features corresponding to 1 drug. 
            confidence (d_features) = *The confidence is based 
                on the number of features and is negatively influenced
                by negation term.
        """
        conclusion = '' 
        l_features = ['operation','target','freq_nr', 'freq_unit', \
                      'strength_nr', 'strength_unit', 'route', 'duration_nr', \
                      'duration_unit', 'dosage_nr']
        for val in l_features:
            if val != 'target':
                if d_features[val] != None:
                    o = d_features[val]
                    conclusion += o + (o != '')*' '
            elif target != None:
                conclusion += target + ' ' 
        if d_features['continue'] == True and d_features['change'] != True:
            conclusion = 'geen verandering'
        if (conclusion == ' ' or (d_features['confidence']<0.15) or 
            (d_features['hypothetical']==True)): 
            conclusion = 'geen voorschrift' 
        return conclusion
    
    def standardize(self, key):
        """
        This function is concerned with data normalization:
        the extracted features, that imply a configuration in 
        regard to treatment, are converted to higher dimensional 
        categories. 

        Conversion examples (in dutch):
        - Voortzetten -> Continue
        - Continueren -> Continue
        - Zo door -> Continue
        """
        if key != None:
            for dimension in self.d_correct:
                regexp = re.compile(r'|'.join(self.d_correct[dimension]))
                if regexp.search(key):
                    return dimension
            return key
        else :
            return None
    
    def fillDictionary(self, l_capture, l_labels, target):
        """
        This function collects all of the features associated with
        the target (a.k.a. the drug). Only one value is allowed per
        feature for each target. The dictionary is only updated with 
        values that aren't equal to None or False. 

        The operation (a.k.a. action) that represents any
        configuration in the prescribed medication relative to the 
        current medication trajectory. (for example: Start, stop, 
        increase and decrease of dosage)

        The Confidence is also calculated with the function 
        calculateConfidence() and a Conclusion is assembled based
        on the collected features with the function 
        calculateConfidence(). Finally, the composed dictionary is
        returned.

        Variables:
            l_labels = list consisting of the labels present in the
                context of the drug. (for example: negation,
                    strength, frequency etc..)
            l_capture = collects the features associated with
                the contextual labels per drug 
                (for example: 15 mg, weekly dose)
            target = drug of interest
        """
        d_features = {'freq_nr' : None, 'freq_unit' : None, 'strength_nr' : 
            None, 'strength_unit': None, 'route': None, 'operation': None, 
            'continue' : None, 'hypothetical' : None, 'change' : None,
            'duration_nr' : None, 'duration_unit': None, 'form': None, 
            'dosage_nr' : None, 'negation' : None, 
            'confidence' : None, 'conclusion' : None}
        for x in l_capture:
            d_sent = literal_eval(x)
            d_sent.update({k:v for k,v in d_features.items() \
                                if v not in [None, False]})
            d_features.update(d_sent)
        d_features['operation'] = self.standardize(d_features['operation'])
        d_features['hypothetical'] = ('hypothetical' in l_labels)
        d_features['continue'] = ('continue' in l_labels)
        d_features['change'] = ('change' in l_labels)
        d_features['negation'] = ('probable_negated_existence' in l_labels)
        d_features['confidence'] = self.calculateConfidence(d_features)
        d_features['conclusion'] = self.generateConclusion(d_features, target)
        return d_features
    
    def extractFeatures(self, report):
        """
        This function first chunks the reports on newline and semi-
         colon characters and then processes each report with 
        the targets and modifiers provided (processReport()).
        The function specifies the absence of a drug, when none is
        found.

        Variables:
            l_entry = list consisting of all the features per 
                drug.
            lbls = list consisting of the labels present in the
                context of the drug. (for example: negation,
                    strength, frequency etc..)
            l_capture = collects the features associated with
                the contextual labels per drug 
                (for example: 15 mg, weekly dose)
            target = drug captured from report
        """
        l_entry = []
        lbls = []
        l_capture = []
        target = 'None'
        report = report.replace('^', '|').replace(';', '|')
        for sentence in report.split('|'):
            try:
                context = self.readContext(sentence)
            except:
                print('--------------------BREAAKK ------------------')
                break
            if len(context.getSectionMarkups()) != 0:
                l_entry, lbls, l_capture, target = self.processReport(l_entry, 
                                        lbls, l_capture, context, target)  
        if l_entry == []: 
            l_entry.append([None]*17)
            l_entry[0][16] = 'geen medicatie'
        return l_entry
    
    def processReport(self, l_entry, lbls, l_capture, context, target):
        """ 
        This function extracts the modifiers and the targets from
        the report. PyContext features the extracted contextual
        labels in the category section and the extracted features
        in the capture section.
        
        All of the features are collected in a dictionary with 
        the function fillDictionary(). These features are appended
        to the list l_entry which consists of all the drugs 
        mentioned in the entry. An updated version of the list is
        returned.

        Variables:
            l_entry = list consisting of all the features per 
                drug.
            lbls = list consisting of the labels present in the
                context of the drug. (for example: negation,
                    strength, frequency etc..)
            l_capture = collects the features associated with
                the contextual labels per drug 
                (for example: 15 mg, weekly dose)
            context = the pyContext rule based object 
            target = drug captured from report
        """
        r_capture = r'<capture>\s(.*)\s</capture>'
        r_category = r'<category>\s\[\'(.*)\'\]\s</category>'
        for section in context.getSectionMarkups():
            for line in str(section).split('\n'):
                if 'MODIFIED BY:' in line:
                    lbls.extend(re.findall(r_category, line))
                    l_capture.extend(re.findall(r_capture, line))
                elif 'TARGET:' in line:
                    target = re.findall(r'<phrase>\s(.*)\s</phrase>', line)
                    lbls.extend(['probable_existence'])
                elif '********************************' in line or \
                    '__________________________________________' in line:
                    d_features = self.fillDictionary(l_capture, lbls, target[0])
                    if (target) != 'None':
                        l_entry.append([target[0], d_features['strength_nr'], 
                                        d_features['strength_unit'], 
                                        d_features['freq_nr'], 
                                        d_features['freq_unit'], 
                                        d_features['route'], 
                                        d_features['duration_nr'],
                                        d_features['duration_unit'],
                                        d_features['dosage_nr'],
                                        d_features['form'],
                                        d_features['operation'], 
                                        d_features['continue'],
                                        d_features['hypothetical'],
                                        d_features['change'],
                                        d_features['negation'],
                                        d_features['confidence'],
                                        d_features['conclusion']
                                        ])
                    lbls = []
                    l_capture = []
                    target = 'None'
        return l_entry, lbls, l_capture, target
        
    def readContext(self, report, print_res=0):
        """
        Creates a context object and reads the report with
        TextBlob. The text from the report is first converted 
        to lowercase. All of the modifiers and targets are 
        marked in each sentence (markup_sentence()) and 
        are then appended to the context object. This context 
        object is returned as output.

        The annotation is printed to the screen 
        if specified (print_res=1). """
        context = pyConText.ConTextDocument()
        blob = TextBlob(report.lower())
        rslts = []
        
        for s in blob.sentences:
            m = self.markup_sentence(s.raw)
            rslts.append(m)
        if print_res == 1:
            print(rslts)
        for r in rslts:
            context.addMarkup(r)
        return context
    
    def createSimpleNetwork(self, sentence):
        """ 
        This function generates a simple network plot that
        visualizes the interaction between modifiers and 
        the targets. In other words: the relation extraction & 
        entity linking steps are exposed.

        Variables:
        Sentence = string / phrase to be processed 
        """ 
        context = self.readContext(sentence)
        plt.figure(figsize=(14,14)) 
        g = context.getDocumentGraph()
        mapping2 = {}
        for new_label, old_label in enumerate(g.nodes()):
            target = re.findall(r'<phrase>\s(.*)\s</phrase>', str(old_label))
            mapping2[new_label] = str(target[0])
        H = nx.convert_node_labels_to_integers(g)
        H = nx.relabel_nodes(H, mapping2)
        
        pos=nx.spring_layout(H)
        nx.draw_networkx(H,pos, node_color="skyblue", node_size=500, 
                         font_size=12, node_shape="s", with_labels=True, 
                         font_weight='bold', linewidths=40)
        return plt
        