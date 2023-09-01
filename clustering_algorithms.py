import random
import math
import pandas as pd
import numpy as np

#Implementation of K means 
#Param: vectors -> list of data points to cluster
#       k -> Number of Clusters
#returns: List of Clusters: ListClusters
def kmeans(vectors: list, k=10) -> list:
    # todo: implement k-means here!
    
    # size of image in terms of vectors
    vector_list_size = len(vectors)

    #initialization of random k clusters
    initialListClusters=np.array([[0,0,0]]*k)
#      print("starting execution initial cluster list ",initialListClusters)
    for i in range(0,k):
        point_index=random.randint(0,vector_list_size-1)
        initialListClusters[i]=vectors[point_index]
#     print("initial cluster centroids: ",initialListClusters)

    # Variable: ListClusters -> stores k list of points which will store the center points after calculation
    ListClusters=np.array([[0,0,0]]*k)
#     print("cluster List to work with ",ListClusters)

    # Variable: ListPointToCluster -> Stores the list of cluster numbers corresponding to each points
    # here the cluster number will be stored at the same index at which the point will be stored in vectore list
    ListPointToCluster=[0]*vector_list_size
#     print('clusters list',ListPointToCluster)
    
    #Loops until initial clusters centers(or you can say the list of cluster in old itiration) matches 
    #the list of cluster points 
    while (ListClusters != initialListClusters).any():
        
        # Variable: 
        minDistance = 0
        
        #check if its the first loop or the corresponding loops
        
        #if the first loop is done then variable ListClusters will already have some center values which 
        #would have been calculated before in the previous iteration in this the initialListClusters will be updated
        #to ListClusters 
        
        # if its the first loop initialListClusters wont change
        if(ListClusters.any()):
#             print("looping continuing...")
            initialListClusters[:]=ListClusters
            centerPoint=ListClusters
        else:
#             print("first loop...")
            centerPoint=initialListClusters
            loopNumber=0
            
        #for each point calculate the distance from center point and calculate new centers based on average
        for i in range(0,vector_list_size):
            
            # Variable listDistanceFromCluster -> stores list of distances between point and the cluster center
            listDistanceFromCluster=[]
            
            # for this point consider each cluster center and find the distance between point and center
            for j in range(0,k):
                distanceValue= np.linalg.norm(centerPoint[j] - vectors[i])
                listDistanceFromCluster.append(distanceValue)
                
            # finding the distance value which is minimum
            minDistance=min(listDistanceFromCluster)
            
            # Variable: nearestCluster -> stores the cluster index which is at minimum distance
            nearestCluster=listDistanceFromCluster.index(minDistance)
            
#             print("current point no: ",i)
#             print("current point r,g,b values: ",vectors[i])
#             print("current center point : ",centerPoint[j])
#             print("list of distances: ",listDistanceFromCluster)
#             print("nearest cluster: ",nearestCluster)

#             Store the cluster number to which this point is nearest to the List at the same index 
#             at which point is present in vector list
            ListPointToCluster[i]=nearestCluster
            
#         After each point is assigned cluster you can check list of cluster values
#         print(ListPointToCluster)
        
        # Calculating new center points based on new cluster points
        # for each cluster
        for clusterNo in range(0,k):
            
            # Variable: totalPointsInCluster -> stores total number of points in cluster to use in average calculation
            # Variable: sumOfClusterPoints -> Stores the sum of all [x,y,z] co-ordinates
            totalPointsInCluster=0
            sumOfClusterPoints = [0,0,0]
            
            # for each point in list
            for p in range(0,len(ListPointToCluster)):
                
                # if the point is assigned to the cluster considered
                if ListPointToCluster[p] == clusterNo:
                    
                    # increment the number of points in cluster and add it to sum
                    totalPointsInCluster = totalPointsInCluster + 1
                    sumOfClusterPoints[0] = sumOfClusterPoints[0] + vectors[p][0]
                    sumOfClusterPoints[1] = sumOfClusterPoints[1] + vectors[p][1]
                    sumOfClusterPoints[2] = sumOfClusterPoints[2] + vectors[p][2]
                    
            # if there are no points in the cluster then change the center randomly
            # else calculate mean and assign new cluster center based on points in that cluster
            if totalPointsInCluster == 0:
                    randomcenterpoint=random.randint(0,vector_list_size-1)
                    ListClusters[clusterNo]= np.array(vectors[randomcenterpoint])
            else:
                # Variable: newClusterAverage -> stores the average values for all 3 co-ordinates of the center point
                newClusterAverage=[0,0,0]
                #for all three co ordinates find average value
                for clusterAvgLoopPointer in range(0,3):
                    newClusterAverage[clusterAvgLoopPointer]=sumOfClusterPoints[clusterAvgLoopPointer] / totalPointsInCluster
                    ListClusters[clusterNo][clusterAvgLoopPointer]=newClusterAverage[clusterAvgLoopPointer]

#         print("new center points for Clusters at end of this loop are \n: ",ListClusters)
#         print("old center points to consider as initial points in while condition will be",initialListClusters)
    
    return ListPointToCluster


# Function ExpandCluster() -> assigns cluster Ids to each point
# Params: dataPoints -> set of data points to use while clustering
# Params: currentPoint -> Current Data Point
# Params: currentpointIndex -> Index at which the current point is present in the Data set
# Params: ListClusters -> List of Clusters assigned to each point
# Params: ClusterID -> Cluster id to use in current iteration
# Params: epsilon -> The value for epsilon decides the distance to which the data points need to be present to current point to be #called as neighbour of current point
# Params: minpts -> The value for minpts decides how many point to be there in the epsilon neighbourhood of the point so as to be #called as core point
# Returns: ListClusters -> List of cluster ids for all points
def ExpandCluster(dataPoints: list, currentPoint: list ,currentpointIndex: int, ListClusters: list, ClusterID: int, epsilon: int, minpts: int):
    
    # Variable: seeds -> Stores the List of Indexes of Points which are neighbours of current point
    seeds = getEpsilonNeighbours(currentpointIndex,dataPoints,epsilon)
#     print("no of seeds found ",len(seeds))
    
    # Check if the lenght of list seeds is less than minpts or not
    # if lenght seeds is greater than it means it is not a core point
    if(len(seeds)<minpts):
#         print("current point ",currentPoint)
#         print("current point index ",dataPoints.index(currentPoint))
#         currentPointPosition=dataPoints.index(currentPoint)
        # Set the cluster id for the current point as -1
        ListClusters[currentpointIndex]= -1
        
        # returns False -> means the current point is not core point
        # returns ListClusters  -> Updated list of cluster ids
        return False,ListClusters
    
    # for each index point in seeds update its cluster id to current cluster id given as parameters to the function
    for seedpoints_index in seeds:
#         print("seed point is ",dataPoints[seedpoints_index])
#         print("seed point is at index ",seedpoints_index)
        ListClusters[seedpoints_index]=ClusterID
#         print("seeds loop updated: ",ListClusters)
        # if the seed point is current point just remove it from seed list or else it will go in infinite loop below by again calculating the neighbours for current point 
        if(seedpoints_index == currentpointIndex):
            seeds.remove(seedpoints_index)
    
    # Untill there are values in seed you check the nieghbours of that point as well
    while(len(seeds)!= 0):
        
        # for each point index in seed
        for seedpoints_index in seeds:
            
            # Get neighbours of the Point in seed
            neighboursofSeedPoint =getEpsilonNeighbours(seedpoints_index,dataPoints,epsilon)
#             print("all neighbours of seed point are ",neighboursofSeedPoint)
            
            # check the no of neighbours found is greater than or equal to minpoint
            # If the above condition is true the neighbour point is also core points and also needs to be assigned in cluster
            if len(neighboursofSeedPoint) >= minpts:
                
                # for each point in neighbouring of seed point
                for indexOfseedPointsNeightbour in neighboursofSeedPoint:
                    
                    # gets the cluster value for the point
                    clusterValueForSeedPointNeighbour = ListClusters[indexOfseedPointsNeightbour]
                    
                    # check if its noice or unclassified
                    if clusterValueForSeedPointNeighbour in [-1,0]:
                        
                        # if the neighbour is unclassified then update it and add it to same cluster and also add it to seed to check if its neighbours are also a part of cluster
                        if clusterValueForSeedPointNeighbour == 0:
                            seeds.append(indexOfseedPointsNeightbour)
                        ListClusters[indexOfseedPointsNeightbour]=ClusterID
#                         print("seed neighbour updated list to ",ListClusters)
                        
            # remove the point from seed
            seeds.remove(seedpoints_index)
                        
    return True,ListClusters

# Function: getEpsilonNeighbours() -> Calculates the neighbours  within epsilon distances
# Params: currentPointIndex -> Current Data point
# Params: dataPoints -> set of data points to use to calculate neighbours of current point
# Params: epsilon -> The value for epsilon decides the distance to which the data points need to be present to current point to be called as neighbour of current point
# Returns: IndexOfPointsDistanceInsideEpsilon -> List of indexes of all points present in epsilon neighbourhood of current point
def getEpsilonNeighbours(currentPointIndex: int, dataPoints: list, epsilon: int): 
    
    # Variable: IndexOfPointsDistanceInsideEpsilon -> List of indexes of all points present in epsilon neighbourhood of current point
    # each index represents the index at which the point is present within dataset
    IndexOfPointsDistanceInsideEpsilon=[]
#     print("inside getEpsilonNeighbours: current point: ",dataPoints[currentPointIndex])

    # for each point in dataset calculate the distance of the point from the current point
    for i in range(0,len(dataPoints)):
        
        # Variable: distance -> stores the distance bettween current point and data point
        distance = np.linalg.norm(dataPoints[currentPointIndex] - dataPoints[i])
        
        # if distance is less then epsilon value add the index of the point to the list
        if(distance < epsilon):
            IndexOfPointsDistanceInsideEpsilon.append(i)
#     print("inside getEpsilonNeighbours: neighbour points found at indexes : ",IndexOfPointsDistanceInsideEpsilon)
    
    # return list of index of all nearest point
    return IndexOfPointsDistanceInsideEpsilon


# Function: generateIDS() -> Generates list of Values of given size to use as cluster ids
# Params: size -> Size of List to be generated
# Returns: ListIds -> List of generated values
def generateIDs(size: int):
    
    # Variable: ListIds -> Stores the List of Generated IDs
    ListIds=[i for i in range(1,size+1)]
    
    #Returns List of ids
    return ListIds

# Function: dbscan() -> Calculates Clusters for given set data points
# Params: vectors -> Set of Data Points to cluster
# Params: minpts -> The value for minpts decides how many point to be there in the epsilon neighbourhood of the point so as to be called as core point
# Params: epsilon -> The value for epsilon decides the distance to which the data points need to be present to current point to be called as neighbour of current point
# Returns: ListClusters -> List of Clusters 
def dbscan(vectors: list, minpts: int, epsilon: int) -> list:
    # todo: implement dbscan here!
    
     # Variable: ListClusters -> stores k list of points which will store the center points after calculation
    ListClusters=[0]*len(vectors)
    
    # Variable: ClusterIDs_iterVar -> List of Ids that can be used to define clusters
    # if your image is too big increase the size in function call 
    if(len(vectors) > 500):
        ClusterIDs_iterVar=iter(generateIDs(20000))
    else:
        ClusterIDs_iterVar=iter(generateIDs(100))
    
    # Variable: currentClusterID -> store the current cluster id
    # Set the next value in list to current cluster id
    currentClusterID=next(ClusterIDs_iterVar)
#     print("cluster id before execution",currentClusterID)
    
    # for each point in data set 
    # if its unclassified than call ExpandCluster() to assign it a cluster id
    for i in range(0,len(vectors)):
        
        # Variable: store the current data point you are working on from the dataset
        currentObject=vectors[i]
        
        # Check if the current data apoint unclustered
        if(ListClusters[i]==0):
            
            # Variable: Updated -> stored Boolean value returned by Expand cluster
            # Updated = True means the data point is assigned to a cluster 
            # Updated = False means the data point was assigned as a noise
            
            Updated,ListClusters=ExpandCluster(vectors,currentObject,i,ListClusters,currentClusterID,epsilon,minpts)
            
            # Updated value is True you need to Update the cluster id for the next iteration
            if(Updated):
#                 print("Clusters Updated as ",ListClusters)
                currentClusterID = next(ClusterIDs_iterVar)
#                 print("and cluster id is also updated to",currentClusterID)
#             else:
#                 print("Clusters Updated for noice values as ",ListPointToCluster)

    
    # Prints List of Cluster
#     print("list clusters",ListClusters)
    
    # return the clusters
    return ListClusters

