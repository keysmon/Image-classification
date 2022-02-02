from keras.datasets import cifar10 # loading data purpose
from random import sample # take sample from the training set as validation set
import random
import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.dim = x.size
order = 1
n_sample = 200
n_of_features = 3072
k_fold = 5

# calculate the euclidean distance between two images
def euclidean_distance(row1,row2):
    return np.linalg.norm(row1-row2,ord = order)

# recusive finding clusters method
def find_cluster(points,centroids,k):
    
    
    clusters = [[] for i in range(k)]
    # recluster points
    for point in points:
        distances = np.array([None]*k)
        # find the distance between a point and all centroids
        for i in range(k):
            centroid = centroids[i]
            distances[i] = euclidean_distance(centroid.x,point.x)
            
        # find the index of cluster it belongs
        belonged_cluster = np.argmin(distances)
        # append the point to the cluster
        clusters[belonged_cluster].append(point)
    
    
    # find the mean of the new cluster
    new_centroids = [[] for i in range(k)]
    for index,cluster in enumerate(clusters):
    
        # find the new cluster center
        x_means = []
        for i in range(n_of_features):
            x_list = [point.x[i] for point in cluster]
            #print(x_list)
            x_mean = sum(x_list)/len(x_list)
            #print(x_mean)
            x_means.append(x_mean)
        
        new_centroids[index] = Point(np.array(x_means),0)
    
    # find whether the final clusters are found
    convergence = True
    for old,new in zip(centroids,new_centroids):
        if not np.array_equal(old.x,new.x):
            convergence = False
            break
    '''
    for centroid in centroids:
        print(centroid.x,centroid.y)
    for centroid in new_centroids:
        print(centroid.x,centroid.y)
    print("\n\n")
    '''
    if convergence == True:
        return new_centroids, clusters
    

    
    return find_cluster(points,new_centroids,k)
    
def cross_validation(trainX,trainY,testX,testY,k):


    
    print("K_Fold of",k)
    splited_trainX = np.array_split(trainX,k_fold)
    splited_trainY = np.array_split(trainY,k_fold)
    match_unmatch = [[0,0] for i in range(k_fold)]
    acc_list = []
    for i in range(k_fold):
    
        # splited training set into training set and validation set
    
        # get training set and validation set
        train_chunks_x = splited_trainX[:i] + splited_trainX[(i+1):]
        train_chunks_y = splited_trainY[:i] + splited_trainY[(i+1):]
        train_chunks_x = np.vstack(train_chunks_x)
        train_chunks_x = train_chunks_x.reshape((-1,n_of_features))
        train_chunks_y = np.vstack(train_chunks_y).flatten()
        #print("train_chunks_x: ",train_chunks_x)
        #print("train_chunks_y: ",train_chunks_y)
        validation_x = splited_trainX[i]
        validation_y = splited_trainY[i]
        #print("Cross validation #",i)
        
        # find accuracies
        acc = k_mean(train_chunks_x,train_chunks_y,validation_x,validation_y,k)
        acc_list.append(acc)
        print("Iteration",i+1,":",acc*100,"%")
        
    # return the avg accuracy
    return sum(acc_list)/len(acc_list)
        


def k_mean(trainX,trainY,testX,testY,k):
    points = []
    for x,y in zip(trainX,trainY):
        points.append(Point(x,y))
    
    
    # select k random image as centroids for clusters
    centroids = sample(points,k)

    final_centroids,clusters = find_cluster(points,centroids,k)
    
    
    # find the majority of class
    for i, cluster in enumerate(clusters):
        y_list = [point.y for point in cluster]
        y_mode = (max(set(y_list), key = y_list.count))
        final_centroids[i].y = y_mode
    
    # find all predicted y values for the testing set
    predicted = []
    for x in testX:
        
        distances = []
        for centroid in final_centroids:
            distances.append(euclidean_distance(x,centroid.x))
        
        label = np.argmin(np.array(distances))
        predicted.append(final_centroids[label].y)
        
        

    

    match = 0
    unmatch = 0
    for p,q in zip(predicted,testY):
       
        if p == q[0]:
            match += 1
        else:
            unmatch += 1
   
    return match/(match+unmatch)

def main():
    
    (trainX,trainY),(testX,testY) = cifar10.load_data()
   
    # normalize data
    trainX = trainX/255
    testX = testX/255
    trainX = np.reshape(trainX,(50000,32*32*3))
    testX = np.reshape(testX,(10000,32*32*3))
  
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    
    
    ks = [3,5,7,11]
    accuracy_l1 = [0,0,0,0]
    accuracy_l2 = [0,0,0,0]
    print("L1 Norm")
    for i in range(4):
        accuracy_l1[i] = cross_validation(trainX,trainY,testX,testY,ks[i])
        print("Accuracy at k of",ks[i],"is",accuracy_l1[i]*100,"%\n")
        
    
    order = 2
    print("L2 Norm")
    for i in range(4):
        accuracy_l2[i] = cross_validation(trainX,trainY,testX,testY,ks[i])
        print("Accuracy at k of",ks[i],"is",accuracy_l2[i]*100,"%\n")
    if max(accuracy_l1) < max(accuracy_l2):
        best_of_norm = 2
        best_of_k = ks[accuracy_l2.index(max(accuracy_l2))]
    else:
        best_of_norm = 1
        best_of_k = ks[accuracy_l1.index(max(accuracy_l1))]

    order = best_of_norm
    
    #print(trainX[:100][:,:n_of_features],trainY[:100],testX[:100][:,:n_of_features],testY[:100])
    print("K_mean model works best at k of",best_of_k,"with L",best_of_norm,"and the accuracy is",100*k_mean(trainX,np.vstack(trainY).flatten(),testX,testY,best_of_k),"%")
    
    
    plt.plot(ks,accuracy_l1,marker="o",label = "norm L1")
    plt.plot(ks,accuracy_l2,marker="o",label = "norm L2")
    plt.legend()
    plt.xlabel('Value of k used in k_mean')
    plt.ylabel('Accuracy on Validation Set')
    plt.axis([0,15,0,1])
    plt.show()
    
    #k_mean(trainX,trainY,testX,testY,k)
    return
    
    
if __name__ == "__main__":
    main()
