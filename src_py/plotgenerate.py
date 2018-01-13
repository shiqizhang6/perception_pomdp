import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def plot_irrelevant(filename,xaxis,file_extension):

    df = pd.read_csv(filename).drop(['Unnamed: 0'],axis=1)
    print df
    fig=plt.figure(figsize=(8,3))

  
    for count,metric in enumerate(list(df)):
        ax=plt.subplot(1,len(list(df)),count+1)

        #l1 = plt.plot([0,1,2,3,4,5],df.loc['pomdp-irrelevant0':'pomdp-irrelevant5',metric],marker='o',linestyle='--',label='MOMDp-irrelevant')1
        l1 = plt.plot(xaxis,df.loc[xaxis[0]:xaxis[-1],metric],marker='o',linestyle='--',label='MOMDp-irrelevant')
        
        plt.ylabel(metric)
        plt.xlim(xaxis[0]-0.5,xaxis[-1]+0.5)
        xleft , xright =ax.get_xlim()
        ybottom , ytop = ax.get_ylim()
        ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)
        plt.xlabel('Number of irrelevant Properties')

    ax.legend(loc='upper left', bbox_to_anchor=(-2.00, 1.35),  shadow=True, ncol=2)      # This is for irrelevant
    fig.tight_layout()
    plt.show()
        #fig.savefig('Results_'+str(num_trials)+'_trials_'+str(num_props)+'_queries_ask_cost'+str(ask_cost)+'_max_cost_prob65.png')
    fig.savefig(filename.split('.')[0]+'.'+file_extension)

def plot_all_strategies(filename,xaxis,file_extension):

    df = pd.read_csv(filename).drop(['Unnamed: 0'],axis=1)
    print df
    fig=plt.figure(figsize=(8,3))

    #Creating plots for different planners and three predicates
    for count,metric in enumerate(list(df)):
        ax=plt.subplot(1,len(list(df)),count+1)

        
        l1 = plt.plot(xaxis,df.loc[0:2,metric],marker='*',linestyle='-',label='MOMDP(ours)')
        l2 = plt.plot(xaxis,df.loc[3:5,metric],marker='D',linestyle=':',label='Predefined')
        l3 = plt.plot(xaxis,df.loc[6:8,metric],marker='o',linestyle='--',label='Predefined Plus')
        l4 = plt.plot(xaxis,df.loc[9:11,metric],marker='^',linestyle='-.',label='Random')
        l5 = plt.plot(xaxis,df.loc[12:14,metric],marker='x',linestyle='-.',label='Random Plus')
   
     
        plt.ylabel(metric)
        plt.xlim(xaxis[0]-0.5,xaxis[-1]+0.5)
        xleft , xright =ax.get_xlim()
        ybottom , ytop = ax.get_ylim()
        ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


        plt.xlabel('Number of Properties')
        




    ax.legend(loc='upper left', bbox_to_anchor=(-3.10, 1.25),  shadow=True, ncol=5)
    
    fig.tight_layout()
    plt.show()
        #fig.savefig('Results_'+str(num_trials)+'_trials_'+str(num_props)+'_queries_ask_cost'+str(ask_cost)+'_max_cost_prob65.png')
    fig.savefig(filename.split('.')[0]+'.'+file_extension)
    
   





def main(argv):
    file_extension='pdf'
    filename='Results_200_constant_props_2pomdp_irrelevants_ask_cost_100.0_action_set_same_as_states_prob65.csv'
    xaxis=[0,1,2,3,4,5]
    #    plot_irrelevant(filename,xaxis,file_extension)

    filename2='Results_200_ask_cost_150_max_cost_150_all_strategies.csv'
    xaxis2=[1,2,3]
    plot_all_strategies(filename2,xaxis2,file_extension)







if __name__ == "__main__":
    main(sys.argv)
