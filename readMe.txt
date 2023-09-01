#Pointers:
There exist following files in the project:
1. notebook.ipynb 
    The above notebook contains the code for following
    a. To import all the necessary modules:
    b. To read the image and convert it to List of points
    c. Calls for the functions of k-means and DBSCAN algorithms to form clusters
    d. To generate the image from the clusters made by algorithms
2. clustering_algorithms.py
    This python file contains following functions
    a. k_means() - Implements K-Mean Algorithm
    b. generateIDs() - Creates a list of values to use as cluster ids in dbscan algorithm
    c. dbscan() - Implements Dbscan Algorithm
    d. ExpandCluster() - Used in Dbscan algorithm to process each point and assign them clusters
    e. getEpsilonNeighbours() - USed for Dbscan to calculate the neighbour points for any point

#Installation:
The following packages have been used in the file:
1. notebook.ipynb
    a. PIL 
    b. numpy
    c. matplotlib.pyplot
2. clustering_algorithms.py
    a. Random
    b. math
    c. Pandas

#Execution:
Steps to execute the following scripts:
1. Place the image to use for clustering in the image folder and update the image name in path accordingly
2. Run cells line wise for k means implementation
3. For dbscan, just comment the following lines for k-means and uncomment for dbscan
    a. image_path
    b. function call
    c. function call for converting clusters to image