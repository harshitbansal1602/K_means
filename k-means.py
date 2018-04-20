import numpy as np
import matplotlib.pyplot as plt 
import sys

if len (sys.argv) != 3:
    print "Usage: python train.py [filename.txt] [no_of_clusters]"
    sys.exit (1)
filename = sys.argv[1]
no_of_clusters = int(sys.argv[2])
data = np.loadtxt(filename)

def initiate_centroids(data,no_of_clusters):

	high = np.max(data)
	low = np.min(data)
	centroids = np.random.randint(low,high,size=(no_of_clusters, data.shape[1])).astype(float)
	return centroids

def assignment(centroids,data):
	
	distances = np.empty((centroids.shape[0],data.shape[0]))
	for i in range(len(centroids)):

		distance_from_i = np.sum((data-centroids[i])**2, axis=1)
		distances[i] = distance_from_i

	assignment = np.argmin(distances, axis=0)

	return assignment

def update_centroids(assignments,centroids):

	for i in range(centroids.shape[0]):
		points = data[np.where(assignments == i)]
		if points.size != 0:
			centroids[i] = np.mean(points,axis = 0)
	return centroids


if __name__ == '__main__':
	
	centroids = initiate_centroids(data,no_of_clusters)
	delta = 99999
	while delta > .01:	
		
		assignments = assignment(centroids, data)
		old_centroids = np.copy(centroids)
		centroids = update_centroids(assignments,centroids)
		delta = np.abs(np.sum(old_centroids - centroids))
	
	#Making plots only if the dimension is 2 
	##Note: if you want to see plots for only first 2 dimensions of dataset please remove the following condition, it will still work.
	if data.shape[1] == 2:
		assignments = assignment(centroids,data)
		col_map = ['b','g','r','c','m','y','k','w']
		
		for i in range(no_of_clusters):
			temp = data[np.where(assignments == i)]
			plt.scatter(temp[:,0],temp[:,1],c = col_map[i],edgecolor = 'w')

		for i in range(len(centroids)):
			plt.scatter(centroids[i,0],centroids[i,1],c = col_map[i],edgecolor = 'k')
		plt.show()

	print 'Centroids are:'
	print centroids
	
