# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 21:24:59 2018

@author: Robin
"""

import pandas as pd
import os
import zipfile
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns


os.chdir('C:\\Users\\Robin\\Desktop\\sanFran\\multi-class-text-classification-cnn-rnn')
os.chdir('C:\\Users\\TapperR\\Desktop\\sfc\\multi-class-text-classification-cnn-rnn')


#in this way you open zip
z = zipfile.ZipFile('data/train.csv.zip')
train = pd.read_csv(z.open('train.csv'))



# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986

mapdata = np.loadtxt("sfcc.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]



    
#Get rid of the bad lat/longs
#train = train[1:30000] #Can't use all the data and complete within 600 sec :(
train['Xok'] = train[train.X<-121].X
train['Yok'] = train[train.Y<40].Y
train = train.dropna()
trainP = train[train.Category == 'GAMBLING']
#trainP = train[train.Category.isin(['PROSTITUTION', 'VANDALISM'])]


#Seaborn FacetGrid, split by crime Category
g = sns.FacetGrid(train, col="Category", col_wrap=2, size=5, aspect=1/asp)

#Show the background map
for ax in g.axes:
    ax.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
    

   
#just make one Grid out of them for the category determined in the trainP
gP = sns.FacetGrid(trainP, col="Category", col_wrap=1, size=10, aspect=1/asp)

for ax in gP.axes:
    ax.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
    
gP.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)


pl.savefig('category_density_plot.png')
   

    
#Kernel Density Estimate plot
g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)

pl.savefig('category_density_plot.png')



#Do a larger plot with prostitution only
pl.figure(figsize=(20,20*asp))
ax = sns.kdeplot(trainP.Xok, trainP.Yok, clip=clipsize, aspect=1/asp, n_levels = 40, legend = True)
ax.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
pl.savefig('gambling_density_plot.png')






