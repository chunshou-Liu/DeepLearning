# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:06:42 2019

@author: Susan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#pre processing
rawdata = pd.read_csv('titanic.csv')

x_train = np.array(rawdata.iloc[:,1:][:800])
x_test = np.array(rawdata.iloc[:,1:][800:])
y_train = np.array(pd.get_dummies(rawdata['Survived'][:800]))
y_test = np.array(pd.get_dummies(rawdata['Survived'][800:]))

x = x_train
y = y_train

#variable
Nin, Nh, Nh2, Nout = 6, 3, 3, 2
lr = 0.005  #learning rate
batchsize = 40
m = 3000   #number of gradient descent pass (epoch)
Q4, Q5, Q6 = 0, 0, 0
losslist, test_loss = [], []
errorlist, test_err = [], []

# =============================================================================
#   Model
# =============================================================================

#initialize weight & bias
def init(Nin, Nh, Nh2, Nout):
    #random seed
    np.random.seed(0)    

    #initialize parameter by random value
    #hidden layer 1
    W1 = np.random.randn(Nin, Nh) / np.amax(Nh)
    b1 = np.zeros((1,Nh))
    #hidden layer 2
    W2 = np.random.randn(Nh, Nh2) / np.amax(Nh2)
    b2 = np.zeros((1,Nh2))
    #output layer
    W3 = np.random.randn(Nh2, Nout) / np.amax(Nout)
    b3 = np.zeros((1,Nout))
          
    return W1, b1, W2, b2, W3, b3

def relu(x):
    return np.clip(x, 0, np.inf)

def drelu(x):
    return (x > 0).astype(np.int8)

def sigmoid(x):
    #activation function 
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    #derivative of sigmoid
    return sigmoid(x) * (1 - sigmoid(x))

def dtanh(x):
    return (1 - np.tanh(x)**2)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(predict, labels, e = 1e-12):
    predict = np.clip(predict, lr, 1.-e)
    N = predict.shape[0]
    ce = -np.sum(np.sum(labels *np.log(predict))) / N
    return ce

#standard normalization
def norm(data, mean=0, std=0):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std), mean, std 

#min-max normalization
def minmaxNorm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#loss function
def loss(model, x, y):
    #model's parameter; theta
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

    #Forward propagation
    z1 = (W1.T).dot(x.T) + b1.T
    a1 = sigmoid(z1.T)
#    a1 = relu(z1.T)
#    a1 = np.tanh(z1.T)
    z2 = (W2.T).dot(a1.T) + b2.T
    a2 = sigmoid(z2.T)
#    a2 = relu(z2.T)
#    a2 = np.tanh(z2.T)
    z3 = (W3.T).dot(a2.T) + b3.T
    probs = sigmoid(z3.T) if Nout==1 else softmax(z3.T) 
    
    #Calculate loss
    ce_loss = cross_entropy(probs,y)
    
    return ce_loss

def error(model, x, y):
    pre = predict(model, x)
    r = ((np.count_nonzero(pre - y.reshape((-1,1)))) / y.shape[0]) if Nout==1 else (np.count_nonzero(pre - y[:,1])) / y.shape[0]
    return r


#build deep nueral-network model
def model(x, y, x_val, y_val, Nin, Nh2, lr, m): 
    if Nout==1:
        y = y[:,1]
        y_val = y_val[:,1]
    #Initialize
    W1, b1, W2, b2, W3, b3 = init(Nin, Nh, Nh2, Nout)
    #initialize model as dict
    model = {}
    
    idx = np.arange(len(x))
    #gradient descent
    for i in range(m):
        np.random.shuffle(idx)
        x, y = x[idx], y[idx]
        miniloss, train_error = [], []
        for j in range(0,x.shape[0],batchsize):
            x_b = x[j:j+batchsize]
            y_b = y[j:j+batchsize]
            #forward propagation
            z1 = W1.T.dot(x_b.T) + b1.T
            a1 = sigmoid(z1.T)
#            a1 = relu(z1.T)
#            a1 = np.tanh(z1.T)
            z2 = W2.T.dot(a1.T) + b2.T
            a2 = sigmoid(z2.T)
#            a2 = relu(z2.T)
#            a2 = np.tanh(z2.T)
            z3 = W3.T.dot(a2.T) + b3.T
            probs = sigmoid(z3.T) if (Nout==1) else softmax(z3.T) 
            
            #backpropagation
            out_delta = (probs - y_b.reshape((probs.shape))) / x_b.shape[0]
            
            dW3 = (a2.T).dot(out_delta)
            db3 = np.sum(out_delta, axis=0, keepdims=True)
            
            delta2 = np.dot(W3,out_delta.T) * dsigmoid(z2)
#            delta2 = np.dot(W3,out_delta.T) * drelu(z2)
#            delta2 = np.dot(W3,out_delta.T) * dtanh(z2)
            dW2 = (a1.T).dot(delta2.T)
            db2 = np.sum(delta2.T, axis=0, keepdims=True)
            
            delta1 = np.dot(W2,delta2) * dsigmoid(z1)
#            delta1 = np.dot(W2,delta2) * drelu(z1)
#            delta1 = np.dot(W2,delta2) * dtanh(z1)                  
            dW1 = (x_b.T).dot(delta1.T)
            db1 = np.sum(delta1.T, axis=0, keepdims=True)
         
            #update gradient descent parameter
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3
                        
            #assign new parameters to the model
            model = {'W1': W1, 
                     'b1': b1, 
                     'W2': W2, 
                     'b2': b2,
                     'W3': W3, 
                     'b3': b3}
             
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            print_loss = False
            if (x.shape[0] == 800):
                if(print_loss):
                    print ("Train-Loss after epoch %i: %f, error rate: %f" %(i, loss(model, x_b, y_b), error(model, x_b, y_b)))
                miniloss.append(loss(model, x_b, y_b))
                train_error.append(error(model, x_b, y_b))


        if(print_loss):
            print ("Train-Loss after epoch %i: %f" %(i, loss(model, x, y)))
            print ("Test Loss after epoch %i: %f" %(i, loss(model, x_val, y_val)))
        losslist.append(np.mean(miniloss))
        errorlist.append(np.mean(train_error))
#        losslist.append(loss(model, x, y))
#        errorlist.append(error(model, x, y))
        test_err.append(error(model, x_val, y_val))
        test_loss.append(loss(model, x_val, y_val))
            
    return model

#perdict result
def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = W1.T.dot(x.T) + b1.T
    a1 = sigmoid(z1.T)
#    a1 = relu(z1.T)
#    a1 = np.tanh(z1.T)
    z2 = W2.T.dot(a1.T) + b2.T
    a2 = sigmoid(z2.T)
#    a2 = relu(z2.T)
#    a2 = np.tanh(z2.T)
    z3 = W3.T.dot(a2.T) + b3.T    
    probs = sigmoid(z3.T) if Nout==1 else softmax(z3.T)

    return np.round(probs) if Nout==1 else np.argmax(probs, axis = 1)

# =============================================================================
#   Implement
# =============================================================================
def discuss(node, normflag):
    
    if normflag == 0:
        #select features
        x = x_train
        x_val = x_test
    else:
        #select features
        x = x_train
        x[:,-1], train_mean, train_std = norm(x[:,-1])
        x_val = x_test
        x_val[:,-1], _, _ = norm(x_val[:,-1], train_mean, train_std)

    Nh2 = node
    if Q4==1:
        x = np.delete(x, i, 1)
        x_val = np.delete(x_val, i, 1)
        
    if Q5==1:
        A = rawdata.copy()
        A = pd.concat([pd.get_dummies(A['Pclass']).astype(int),A],axis=1)
        A = A.drop(columns=['Pclass', 'Survived'])
        
        x = np.array(A.iloc[:,1:][:800])
        x_val = np.array(A.iloc[:,1:][800:])
        if normflag== 1:
            x[:,-1], train_mean, train_std = norm(x[:,-1])
            x_val[:,-1], _, _ = norm(x_val[:,-1], train_mean, train_std)

    #build a model with a 3-dimensional hidden layer
    dnnmodel = model(x, y, x_val, y_test, Nin, Nh2, lr, m)
    result = predict(dnnmodel, x)
    
    print("The Network's structure:" , "[", Nin, "-", Nh, "-", Nh2, "-", Nout, "]")
    
    if Q6==1:
        # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        survived = [3, 0, 23, 5, 0, 99]
        dead = [3, 1, 23, 5, 0, 99]
        print("Test data before normalize：\n" + str(rawdata.columns)[6:-17])
        print("survived：" + str(survived))
        print("dead：" + str(dead) + "\n")

        if normflag== 1:
            survived[-1] = (survived[-1]-train_mean)/train_std
            dead[-1] = (dead[-1]-train_mean)/train_std
        survived = np.array(survived).reshape((1,-1))
        dead = np.array(dead).reshape((1,-1))
        pre = np.concatenate([survived, dead], axis=0)
        Q6_result = predict(dnnmodel, pre)
        Ans_s = 'yes' if Q6_result[0]==1 else 'no'
        Ans_d = 'yes' if Q6_result[1]==0 else 'no'
        print("Predict of survived:%s，dead:%s" % (Ans_s,Ans_d))

    if Q4==0 and Q6==0: 
        #loss curve for training
        plt.plot(losslist)
        plt.title("Training Loss")
        plt.xlabel('# of epoch')
        plt.ylabel('cross entropy')
        plt.show()
        plt.close()
    
        #loss curve for testing
        plt.plot(test_loss, color='orange')
        plt.title("Testing Loss")
        plt.xlabel('# of epoch')
        plt.ylabel('cross entropy')
        plt.show()
        plt.close()
        
        #error rate curve for training
        plt.plot(errorlist)
        plt.title("Training Error Rate")
        plt.xlabel('# of epoch')
        plt.ylabel('Error Rate')
        plt.show()
        plt.close()
        
        #error rate curve for testing
        plt.plot(test_err, color='orange')
        plt.title("Testing Error Rate")
        plt.xlabel('# of epoch')
        plt.ylabel('Error Rate')
        plt.show()
        plt.close()

    return result

 #%%
'''Q1 & 2'''
losslist, errorlist, test_loss, test_err = [], [], [], []
lr = 0.005 #目前最好
result2 = discuss(3,0)

 #%%
'''Q3_normalize'''
normflag = 1
losslist, errorlist, test_loss, test_err = [], [], [], []
Q3_result = discuss(3,1)
#%%

'''Q4_most affective feature'''
#variable
Nin, Nh, Nh2, Nout = 5, 3, 3, 2
normflag = 1

for i in range(0,6):
    print(i)
    losslist, errorlist, test_loss, test_err = [], [], [], []
    Q4 = 1
    Q4_result = discuss(3,1)
    print('without %s , Acc = %f' %(rawdata.columns[i+1], 1 - np.array(test_err)[-1]) )

Q4=0
#%%
'''Q5_one-hot'''
Q5 = 1
lr = 0.005
Nin, Nh, Nh2, Nout = 7, 3, 3, 2
losslist, errorlist, test_loss, test_err = [], [], [], []
Q5_result = discuss(3,1)

print('Acc with categorical = %f' %(1 - np.array(test_err)[-1]) )
Q5=0
Nin, Nh, Nh2, Nout = 6, 3, 3, 2
losslist, errorlist, test_loss, test_err = [], [], [], []
Q5_result = discuss(3,1)
print('Acc without categorical = %f' %(1 - np.array(test_err)[-1]) )

#%%
'''Q6_Make predict data'''
Q6=1
lr = 0.005
Nin, Nh, Nh2, Nout = 6, 3, 3, 2
losslist, errorlist, test_loss, test_err = [], [], [], []
Q6_result = discuss(3,1)
Q6=0