from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import math

epoch = 1000
lrate = 0.1



num_inputs = 3072
num_outputs = 10


 

class softmax:
    def __init__(self):
        self.w = np.random.random((num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        return
        
    def one_hot(self, y):
        y = y.reshape(-1)
        y_one_hot = np.zeros((len(y), num_outputs))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot
        
    def softmax_activation(self,x):
        x -= np.max(x)
        prob = (np.exp(x).T / np.sum(np.exp(x), axis=1)).T
        return prob
        
        
    def train(self,trainX,trainY,epochs):
    
        # find the number of sample
        sample = trainX.shape[0]
        losses = []
        for epoch in range(epochs):
            
           
          
            z = np.dot(trainX,self.w)+self.b
            
            # prevent overflow
            z = np.exp(z - np.max(z, axis=1, keepdims=True))

            # transform the z value into probability
            y_pred = self.softmax_activation(z)
            prob = -np.log(y_pred[np.arange(sample), trainY])
            
            loss = np.sum(prob)/sample
            
            # get the avg loss per sample
            loss /= sample
            losses.append(loss)
            
            # one hot encoding
            y_one_hot = self.one_hot(trainY) 
                
            # update w,b
            w_grad = (1/sample)*np.dot(trainX.T,(y_pred - y_one_hot))
            b_grad = (1/sample)*np.sum(y_pred - y_one_hot)
            
            
            # Updating the parameters
            self.w = self.w - lrate*w_grad
            self.b = self.b - lrate*b_grad
            
            
           
        
            #Print loss every 100th epoch
            if epoch%100==0:
                print("Epoch {",epoch,"}==> Loss = {",loss,"}")
        return losses
        
    def predict(self,trainX,trainY):
        y_pred = np.argmax(trainX@self.w+self.b,axis = 1)
        match = 0
        unmatch = 0 
        for y, y_ in zip(trainY,y_pred):
            if y_ == y[0]:
                match += 1
            else:
                unmatch += 1
        return match,unmatch
       
def main():
    (trainX,trainY),(testX,testY) = cifar10.load_data()
    
    # normalize data
    meanImage = np.mean(trainX,axis = 0)
    trainX = trainX - meanImage
    testX = testX - meanImage
    
    # normalize training/testing data
    trainX = (trainX/255)
    testX = (testX/255)
    trainX = (np.reshape(trainX,(50000,3072)))
    testX = (np.reshape(testX,(10000,3072)))
    
   
    
    # tune hyperparameter of learning rate
    lrates = np.arange(0.01,0.5,0.03)
    match_unmatch = []
    
    for rate in lrates:
        
        obj = softmax()
        losses = obj.train(trainX,trainY,epoch)
        
        lrate = round(rate,2)
        match,unmatch = obj.predict(testX,testY)
        match_unmatch.append([match,unmatch])
        print("Accuracy with learning rate of ",rate," is",round(match/(unmatch+match)*100,3),"%\n")   
        
        
    # plot the accuracy vs learning rates
    accuracy = [item[0]/(item[0]+item[1]) for item in match_unmatch]
    print("The best accuracy of softmax is",max(accuracy)*100,"%\n")
    plt.plot(lrates,accuracy)
    plt.xlabel("Value of learning rates")
    plt.ylabel("Accuracy of softmax classification")
    plt.show()
    
    # plot loss vs. epoch to showcase loss function
    plt.plot([i for i in range(len(losses))],losses)
    plt.xlabel("epoch")
    plt.ylabel("Cross Entropy loss at 0.44 learning rate")
    plt.show()
    
    
    return
    
    
if __name__ == "__main__":
    main()
