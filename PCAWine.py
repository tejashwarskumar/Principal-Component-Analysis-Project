import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt

wineData = pd.read_csv("C:/My Files/Excelr/07 - PCA/Assignment/wine.csv")
len(wineData)
wineData.columns
wineData = wineData.iloc[:,1:13]

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#standarize Data
windeData_normal = scale(wineData)
pcaData = PCA(n_components=7)
pca = pcaData.fit_transform(windeData_normal)
var = pcaData.explained_variance_ratio_
pcaData.components_[0]

var1 = nm.cumsum(nm.round(var,decimals = 4)*100)
var1
plt.plot(var1,color="red")

x = pca[:,0]
y = pca[:,1]
z = pca[:2:7]
plt.scatter(x,y,color=["red"])
len(pca)

newDF = pd.DataFrame(pca[:,0:7])

#Clustering
newDF.describe()
newDF.corr()

def normDatafn(i):
    x = i - i.min()/i.max() - i.min()
    return x

normData = normDatafn(newDF)
normData

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(normData,method="complete",metric='euclidean')
sch.dendrogram(z)

#Data is more so kmeans clustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

kmeans = KMeans(n_clusters=4)
kmeans.fit(normData)
kmeans.labels_
k = list(range(2,15))

from sklearn.cluster import KMeans
TWSS = [];
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(normData)
    WSS=[];
    for j in range(i):
        WSS.append(sum(cdist(normData.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,normData.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
    len(WSS)
len(TWSS)
plt.plot(k,TWSS,'ro-');plt.xlabel("K Cluster Number");plt.ylabel("Total Within Square");

model = KMeans(9)
model.fit(normData)
model.labels_
labels = pd.Series(model.labels_)
newDF['clust'] = labels
newDF.columns
newDF = newDF.iloc[:,[7,0,1,2,3,4,5,6]]
newDF.iloc[:,1:7].groupby(newDF['clust']).mean()
