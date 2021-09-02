import re

def is_negated(re_output):
    """ 
    Check whether concept in text is negated.
    
    Input:
        re_output = tuple with groups captured by regex
    Output:
        True / False (boolean) -> depending on free text 
    """
    if re_output[0][0] != '' or re_output[0][2] != '':
        return True
    else : 
        return False

def check_concept_status(sent, query): ## Give right query
    """
    Check the status of the concept. Only return output if the 
    concept is found in the text & if it is not negated!
    
    Input:
        sent = sentence (string)
        query = regular expression with triggerwords
    Output:
        concept_status = whether or not the concept is present
        r_out = list consisting of the collected triggerwords
    
    """
    concept_status = 0
    r_out = ''
    r1 = re.findall(query,sent)
    #print(r1)
    if r1 != []:
        if is_negated(r1) == True:
            concept_status = 0
            r_out = '' # dont show match if empty!
        else :
            concept_status = 1
            r_out = r1#[0][1]
    return concept_status, r_out

    
    
def concept_present(l_concepts, row): ## check 
    """
    Check for concept in the Electronic Medical records
    
    Input:
        l_concepts = list with concepts to look for (this accepts regex!)
        row = row in dataframe
    Output:
        output_list  = list with the intercepted concepts that can be intergrated in the dataframe
    """
    # ild except -> a-Z
    # intertestial pneumonia
    not_prior = ['geen tekenen van', 'geen aanwijzingen voor', 'geen', 'geen kenmerken van', 'geen argumenten voor', ' no', 'no signs of', 'no indication of', 'no indication for', 'dd', 'non-']  # , 'dd', 'mogelijk', 
    not_post = ['uitgesloten', 'onwaarschijnlijk', 'afwezig', 'absent', 'unlikely']

    
    #not_within = 
    
    tot_list = [l_concepts]
    l_lbls = ['Algemeen']
    
    output_list = []
    
    value = row['XANTWOORD']
    try: 
        value = value.lower()
    except:
        return False, '', '', False, '', '', False, '', ''
    
    for ix, l in enumerate(tot_list): 
        #l = ['(?<![A-z])%s(?![A-z])' % str(i) for i in l]
        query = '(' + '|'.join(not_prior) + ')?' + '\s?' + '(' + '|'.join(l) + ')' + '\s?' + '(' + '|'.join(not_post) + ')?'
        #d_query[ix] = query
        name = l_lbls[ix] 
        
        concept, r_out = check_concept_status(value, query)
        if r_out == '':
            output_list.extend([concept, '', ''])
        else :
            output_list.extend([concept, r_out[0][1], [' '.join(s).strip() for s in r_out]])
    return output_list