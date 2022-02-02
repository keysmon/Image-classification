from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

n_of_features = 3072
n_of_labels = 10
k_fold = 5
# calculate the euclidean distance between two images
def euclidean_distance(row1,row2,order):
    return np.linalg.norm(row1-row2,ord = order)

 
def get_majority_class(trainX,trainY,test,k,order):
    distances = []
    for index,row in enumerate(trainX):
        distances.append([index,euclidean_distance(row,test,order)])
    k_neighbors = np.array([0]*n_of_labels)
    
    for i in range(k):
        closest = min(distances,key = lambda input: input[1])
        k_neighbors[trainY[closest[0]]] += 1
        distances.remove(closest)
    return np.argmax(k_neighbors)
    
    
# 5 fold cross validation
def cross_validation(trainX,trainY,k,order):
    splited_trainX = np.array_split(trainX,k_fold)
    splited_trainY = np.array_split(trainY,k_fold)
    match_unmatch = [[0,0] for i in range(k_fold)]
    for i in range(k_fold):
    
        # get training set and validation set
        train_chunks_x = splited_trainX[:i] + splited_trainX[(i+1):]
        train_chunks_y = splited_trainY[:i] + splited_trainY[(i+1):]
      
        train_chunks_x = np.vstack(train_chunks_x)
        train_chunks_x = train_chunks_x.reshape((-1,n_of_features))
        train_chunks_y = np.vstack(train_chunks_y).flatten()
       
        validation_x = splited_trainX[i]
        validation_y = splited_trainY[i]
        
        # find accuracy
        for x,y in zip(validation_x,validation_y):
            if get_majority_class(train_chunks_x,train_chunks_y,x,k,order) == y:
                match_unmatch[i][0] +=1
            else:
                match_unmatch[i][1] +=1
        print("Iteration",i+1, "Matched - Unmatched: ", match_unmatch[i][0],"-",match_unmatch[i][1])
    #for i,item in enumerate(match_unmatch):
        #print("Fold #",i+1,"Matched - Unmatched: ", item[0],"-",item[1]," Accuracy:",round(item[0]/(item[0]+item[1])*100,1),"%")
    matched_total = [i[0] for i in match_unmatch]
    unmatched_total = [i[1] for i in match_unmatch]
    mean =sum(matched_total)/(sum(matched_total)+sum(unmatched_total))
    return mean
        
    




def main():
   
    (trainX,trainY),(testX,testY) = cifar10.load_data()
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    # normalize data
    trainX = (trainX/255)
    testX = (testX/255)
    trainX = (np.reshape(trainX,(50000,3072)))
    testX = (np.reshape(testX,(10000,3072)))
    
    

    ks = [3,5,7,11]
    accuracy_l1 = [0,0,0,0]
    accuracy_l2 = [0,0,0,0]
    
    
   
    
    print("L1 NORM")
    for i in range(4):
        accuracy_l1[i] = cross_validation(trainX,trainY,ks[i],1)
        print("Accuracy at k of",ks[i],"is",round(accuracy_l1[i]*100,1),"%\n")
    print("L2 NORM")
    for i in range(4):
        accuracy_l2[i] = cross_validation(trainX,trainY,ks[i],2)
        print("Accuracy at k of",ks[i],"is",round(accuracy_l2[i]*100,1),"%\n") 
        
    best_of_k_l1 = accuracy_l1.index(max(accuracy_l1))
    best_of_k_l2 = accuracy_l2.index(max(accuracy_l2))
    if max(accuracy_l1)< max(accuracy_l2):
        best_of_norm = 2
        best_of_k = ks[best_of_k_l2]
    else:
        best_of_norm = 1
        best_of_k = ks[best_of_k_l1]
    
    matched = 0
    unmatched = 0
    
    for x,y in zip(testX,testY):
        predicted = get_majority_class(trainX,trainY,x,best_of_k,best_of_norm)
        #print(predicted,y[0])
        
        
        if predicted == y[0]:
           
            matched += 1
        else:
            unmatched += 1
    print("KNN models works best at k of",best_of_k,"with L",best_of_norm,"norm.","and the accuracy is",matched/(matched+unmatched)*100,"%")
    
    
    plt.plot(ks,accuracy_l1,marker = "o",label = "norm L1")
    plt.plot(ks,accuracy_l2,marker = "o",label = "norm L2")
    plt.legend()
    plt.xlabel("Value of k used in knn")
    plt.ylabel("Accuracy on validation set")
    plt.axis([2,12,0,0.5])
    plt.show()


    return
    
    
if __name__ == "__main__":
    main()
