import matplotlib.pyplot as plt
import pandas as pd

def plot_bar(var1, var2) :
    """
    Create a barplot with var1 against var2
    
    Input:
        var1 = variable 1
        var2 = variable 2
    Output:
        bar plot
    """
    plt.rcParams.update({'font.size': 20})

    d = {'cat': var1, 'cluster': var2} 
    df_bar = pd.DataFrame(data=d)
    #df_bar = df_bar.sample(frac=1)
    fig, ax = plt.subplots(figsize=(15,7))
    df_bar.groupby(['cluster', 'cat']).size().unstack().plot(ax=ax, kind = 'bar') 
    plt.legend(loc=1, prop={'size': 16})
    return