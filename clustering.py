# clustering.py
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pingouin as pg
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn import svm
import pdb
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

def creation_training(sheet = "ALL (updated)"):
    df = pd.read_excel('Bilateral ADX DB - 2022.07.25_JB.xlsx', sheet_name = sheet)
    centers = df['Centre'].values
    for i, values in enumerate(centers):
        if values in ["Munich", "Torino", "Taiwan"]:
            centers[i] = 0
        elif values in ["Sendai"]:
            centers[i] = 1
        elif values in ["Brisbane"]:
            centers[i] = 2 
        else:
            centers[i] = 3
    DATA_original = df.iloc[:,[12, 15,14,16,23,24,27,17,18,19,20,21,22,34,35,33,28,9, 77, 78]]
    
    Xmeans1 = np.nanmean(DATA_original.values,axis=0)
    for i in np.arange(Xmeans1.shape[0]):
            X1 = DATA_original.iloc[:,i].values
            X1[np.isnan(X1)] = Xmeans1[i]
            DATA_original.iloc[:,i] = X1
    return DATA_original, centers

DATA_original, centers = creation_training()

class DimReduction():
    def __init__(self, dim_out):
        # load para
        self.model = []
        # define model type
        self.dim_out = dim_out
    def PCA1(self, data_in, to_fit):
        #Apply the PCA to the dataset
        X = preprocessing.scale((data_in-data_in.mean())/data_in.mean())
        if to_fit == 1:
            self.model = PCA(n_components=2)
            self.model.fit(X)
        else :
            return self.model.transform(X)

np.random.seed(0)

reduction = DimReduction(2)
reduction.PCA1(DATA_original.values,1)

data_compressed = reduction.PCA1(DATA_original.values,0) 

for feat in DATA_original.keys():
    var0 = np.var(DATA_original[feat][centers == 0])*sum(centers==0)
    var1 = np.var(DATA_original[feat][centers == 1])*sum(centers==1)
    var2 = np.var(DATA_original[feat][centers == 2])*sum(centers==3)
    var3 = np.var(DATA_original[feat][centers == 3])*sum(centers==3)
    SSW = (var0+var1+var2+var3)/len(centers)
    mean_all = np.mean(DATA_original[feat])
    mean0 = np.mean(DATA_original[feat][centers == 0])
    mean1 = np.mean(DATA_original[feat][centers == 1])
    mean2 = np.mean(DATA_original[feat][centers == 2])
    mean3 = np.mean(DATA_original[feat][centers == 3])
    SSB = ((mean0 - mean_all)**2*sum(centers==0)+(mean1 - mean_all)**2*sum(centers==1)+(mean2 - mean_all)**2*sum(centers==2)+(mean3 - mean_all)**2*sum(centers==3))/len(centers)
    ICC = SSB / (SSB+SSW)
    print(f"Feature: {feat}. ICC: {ICC}.")

fig, ax = plt.subplots(figsize=(6,4))
dim = 30
w = 0.6
plt.grid()
plt.scatter(data_compressed[centers==0,0],data_compressed[centers==0,1], color = '#107f00', s = dim, linewidths = w, edgecolor = 'k', label='Munich, Turin, Taipei City')
plt.scatter(data_compressed[centers==1,0],data_compressed[centers==1,1], color = '#0f7ffe', s = dim, linewidths = w, edgecolor = 'k', label='Sendai')
plt.scatter(data_compressed[centers==2,0],data_compressed[centers==2,1],color = '#fb0307',s = dim, linewidths = w,edgecolor = 'k', label='Brisbane')
plt.scatter(data_compressed[centers==3,0],data_compressed[centers==3,1],color = 'k',s = dim, linewidths = w,edgecolor = 'k', label='Yokohama')

ax.scatter(np.mean(data_compressed[centers==0,0]),np.mean(data_compressed[centers==0,1]),c='k',s=20,marker = 'X')
ax.scatter(np.mean(data_compressed[centers==1,0]),np.mean(data_compressed[centers==1,1]),c='k',s=20,marker = 'X')
ax.scatter(np.mean(data_compressed[centers==2,0]),np.mean(data_compressed[centers==2,1]),c='k',s=20,marker = 'X')
ax.scatter(np.mean(data_compressed[centers==3,0]),np.mean(data_compressed[centers==3,1]),c='k',s=20,marker = 'X')
ellipse2 = Ellipse(xy=(np.mean(data_compressed[centers==0,0]),np.mean(data_compressed[centers==0,1])), width=np.std(data_compressed[centers==0,0])*2, height=np.std(data_compressed[centers==0,1])*2, edgecolor='#107f00', fc='None', lw=1)
ellipse = Ellipse(xy=(np.mean(data_compressed[centers==1,0]),np.mean(data_compressed[centers==1,1])), width=np.std(data_compressed[centers==1,0])*2, height=np.std(data_compressed[centers==1,1])*2, edgecolor='#0f7ffe', fc='None', lw=1)
ellipse1 = Ellipse(xy=(np.mean(data_compressed[centers==2,0]),np.mean(data_compressed[centers==2,1])), width=np.std(data_compressed[centers==2,0])*2, height=np.std(data_compressed[centers==2,1])*2, edgecolor='#fb0307', fc='None', lw=1)
ellipse3 = Ellipse(xy=(np.mean(data_compressed[centers==3,0]),np.mean(data_compressed[centers==3,1])), width=np.std(data_compressed[centers==3,0])*2, height=np.std(data_compressed[centers==3,1])*2, edgecolor='k', fc='None', lw=1)
ax.add_patch(ellipse1)
ax.add_patch(ellipse)
ax.add_patch(ellipse2)
ax.add_patch(ellipse3)
plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
fig.subplots_adjust(bottom=0.2)
plt.savefig( 'PCA.png', dpi = 1200)

