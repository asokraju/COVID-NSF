from os import walk, listdir
import numpy as np

import matplotlib.pyplot as plt
from joblib import dump, load
from scipy.io import savemat, loadmat
import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def list_files2(directory, extension):
    w = []
    mat_file_names = []
    for (dirpath, dirnames, filenames) in walk(directory):
        #print(dirpath, dirnames, filenames)
        w.append(dirpath[-3:])
        #return (f for f in filenames if f.endswith('.' + extension))
        for f in filenames:
             if f.endswith('.' + extension):
                # print(f)
                mat_file_names.append(dirpath+ "/" + f)
    return w[1:], mat_file_names

def df_gen(file_names, index, col_names):
    state_of_exp = []
    for file in iter(file_names):
        data = loadmat(file)
        print(data.keys())
        state = np.array(data['states'])[:,index][:100]
        state = state.tolist()
        state_of_exp.append(state)
    state_of_exp = np.array(state_of_exp).T
    return pd.DataFrame(state_of_exp, columns = col_names) 


def plot(savefig_filename=None):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,15))
    sns.barplot(data=S, ax=axes[0, 0])
    sns.barplot(data=E, ax=axes[0, 1])
    sns.barplot(data=I, ax=axes[1, 0])
    sns.barplot(data=R, ax=axes[1, 1])
    axes[0, 0].set_ylabel('Susceptible', fontsize=15)
    axes[0, 1].set_ylabel('Exposed', fontsize=15)
    axes[1, 0].set_ylabel('Infectious', fontsize=15)
    axes[1, 1].set_ylabel('Removed', fontsize=15)
    if savefig_filename is not None:
        assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = 'pdf')
    else:
        plt.show()


directory = './results/experiment-3/'
extension = 'mat'   
w, mat_file_names = list_files2(directory, extension)   
names = ['S', 'E', 'I', 'R']
S = df_gen(mat_file_names, index =0, col_names = w)
E = df_gen(mat_file_names, index =1, col_names = w)
I = df_gen(mat_file_names, index =2, col_names = w)
R = df_gen(mat_file_names, index =3, col_names = w)
file_name = './results/exp_3.pdf'
plot(savefig_filename=file_name)
# plot()
print(S.head())