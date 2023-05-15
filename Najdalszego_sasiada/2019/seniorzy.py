import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# loading data from Excel
data = pd.read_excel('zmienne.xlsx', sheet_name = 'Seniorzy 2019', usecols = 'A:E', skiprows = 1)


# calculation of the distance matrix and creation of a dendrogram with labels
distances = pdist(data.iloc[:,1:], metric='euclidean')
linkage_matrix = linkage(distances, method='single')
dendrogram(linkage_matrix, labels=data.iloc[:,0].values, orientation = "right")
plt.show()