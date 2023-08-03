#Sarayu Vyakaranam

#import libraries
import csv as c
import numpy as np
import pandas
import matplotlib.pyplot as plt
import warnings
# suppress warnings
warnings.filterwarnings('ignore')

#created a helper function to print shape of df
def df_shape(df):
    rows=df.shape[0]
    cols=df.shape[1]
    print("rows of df ="+str(rows))
    print("cols of df ="+str(cols))

#created a helper function to print shape of column
def y_shape(y):
    rows=y.shape[0]
    print("rows of y ="+str(rows))

#created a helper function to print lists
def print_list(l1):
    for x in l1:
        for y in x:
            print(y)

#creating the entire vocabulary
def create_word_list(df):
    col_list=list(df["text"])
    #col_list=df
    word_list=[]
    for sentence in col_list:
        for word in sentence.split(" "):
            word_list.append(word)
    #print(word_list) which is a list of all unique words
    #vocabulary_unique=set(word_list)
    #vocabulary=list(vocabulary_unique)
    vocabulary=list(word_list)
    #print(vocabulary) #list of unique words
    return vocabulary

#bag of words
def create_bag_of_words(df, vocabulary):
  leng=len(df['text'])
  dict={}
  for i in vocabulary:
      if i not in dict:
          dict[i]=1
      else:
          dict[i]+=1
  return dict

#tokenize a sentence
def create_tokenize(sentence):
    tokenized=[]
    for word in sentence.split(" "):
        tokenized.append(word)
    return tokenized

#create the feature vector 
def create_feature_vector(df,vocabulary, dict):
  feature_vector=[]
  leng=len(df['text'])
  for sentence in range(leng):
      w=create_tokenize(df['text'][sentence])
      dict1={}
      for k in w:
          if k in dict1:
              dict1[k]+=1
          else:
              dict1[k]=1
      temp=[]
      for i in dict:
        if i in dict1:
          temp.append(dict1[i])
        else:
          temp.append(0)
      feature_vector.append(temp)
  return feature_vector

#one hot encoding
def create_y_vector(df):
    leng=(len(df['emotions']))
    feature_vector=[]
    for j in range(leng):
        emotion=df['emotions'][j]
        temp_output=np.zeros(6)
        if emotion=='sadness' :
            temp_output[0]=1
        if emotion=='joy':
            temp_output[1]=1
        if emotion=='love' :
            temp_output[2]=1
        if emotion=='anger':
            temp_output[3]=1
        if emotion=='fear':
            temp_output[4]=1
        if emotion=='surprise':
            temp_output[5]=1
        feature_vector.append(temp_output)
    feature_vector=np.array(feature_vector)
    return feature_vector

#find the output labels 
def output_labels(df):
    labels=[]
    for i in range(len(df['emotions'])):
        a=[0]*6
        emotion=df['emotions'][i]
        if emotion=='sadness':
            labels.append(0)
        elif emotion=='joy':
            labels.append(1)
        elif emotion=='love':
            labels.append(2)
        elif emotion=='anger':
            labels.append(3)
        elif emotion=='fear':
            labels.append(4)
        else:
            labels.append(5)
    labels=np.array(labels)
    return labels

#format result to excel sheet
def format_result(answers, df2):
    vector_output=pandas.DataFrame(answers, columns=['emotions'])
    leng=len(answers)
    #print(leng)
    #print(len(vector_output['emotions']))
    for i in range(len(vector_output['emotions'])):
        emotion=vector_output['emotions'][i]
        if emotion==0:
            vector_output['emotions'][i]='sadness'
        elif emotion==1:
            vector_output['emotions'][i]='joy'
        elif emotion==2:
            vector_output['emotions'][i]='love'
        elif emotion==3:
            vector_output['emotions'][i]='anger'
        elif emotion==4:
            vector_output['emotions'][i]='fear'
        elif emotion==5:
            vector_output['emotions'][i]='surprise'
    df2['emotions']=vector_output
    df2.to_csv('test_lr.csv',index=False)

#training loss
def loss_training(y,z):
    ans=np.multiply(y,np.log(z))
    return -(np.sum(ans)/1200)

#validation loss
def loss_validation(y,z):
    ans=np.multiply(y,np.log(z))
    return -(np.sum(ans)/1200)

#softmax
def softmax(z):
    arr=np.exp(z)
    return np.divide(arr,np.sum(arr,axis=1,keepdims=True))

#selected parameters based on best performance in cross validation
def LR_param_init():
    episodes=600
    trial=250
    alpha=0.6
    return episodes, trial, alpha
 
#separete function for logistic regression
def lr_function(episodes, trial, alpha, feature_vector, y_train, t_feature_vector, weights):
    for i in range(0,episodes):
        mul=np.matmul(feature_vector,weights)
        output=softmax(mul)
        counter=0
        leng=len(feature_vector)
        for counter in range(0,leng,trial):
            a=feature_vector[counter:counter+trial]
            b=output[counter:counter+trial]-y_train[counter:counter+trial]
            t1=np.divide(np.matmul(a.T,b),a.shape[0])
            weights-=alpha*(np.add(t1, 0.001*weights))
            #counter+=trial
    mul=np.matmul(t_feature_vector,weights)
    output=softmax(mul)
    result=np.argmax(output,axis=1)
    return result

#separate function for cross validation
def lr_cross_validation_function1(feature_vector, y_train, labels, mini,p, e, a, t, r, temp_testing_accuracy, temp_training_accuracy, temp_testing_loss, temp_training_loss):
    feature_vector_20=feature_vector[p:(p+mini)]
    feature_vector_80=np.concatenate((feature_vector[:p], feature_vector[(p+mini):]))
    y_train_20=y_train[p:(p+mini)]
    y_train_80=np.concatenate((y_train[:p], y_train[(p+mini):]))
    labels_20=labels[p:(p+mini)]
    labels_80=np.concatenate((labels[:p], labels[(p+mini):]))
    W=np.zeros((feature_vector_80.shape[1], 6))
    for i in range(e):
        featureW = np.matmul(feature_vector_80, W)
        Y_probability=softmax(z=featureW)
        j = 0
        while (j < len(feature_vector_80)):
            x=feature_vector_80[j:min(j+t, 1200)]
            diff=Y_probability[j:min(j+t, 1200)]-y_train_80[j:min(j+t, 1200)]
            xdWeights=np.divide(np.matmul(x.T, diff), x.shape[0])
            W=W-(a * (np.add(xdWeights, r*W)))
            j=j+t
    featureW_20=np.matmul(feature_vector_20, W)
    testProbability=softmax(featureW_20)
    testPrediction=np.argmax(testProbability, axis=1)
    featureW_80=np.matmul(feature_vector_80, W)
    YProbability=softmax(z=featureW_80)
    trainPrediction=np.argmax(YProbability, axis=1)
    temp_testing_accuracy.append((np.sum(labels_20 == testPrediction)/len(testPrediction) * 100))
    temp_training_accuracy.append((np.sum(labels_80 == trainPrediction)/len(trainPrediction) * 100)) 
    temp_testing_loss.append(loss_validation(y_train_20,testProbability ))
    temp_training_loss.append(loss_training(y_train_80,YProbability ))         
    p=p+mini

#function to make plots for both the parts
def make_plots(s, t,choice):
    if choice==1:
        fig, ax = plt.subplots()
        ax.plot(s, t, color='tomato', marker='o',markerfacecolor='darkred', markersize=9)
        ax.set(xlabel='alpha- Learning Rate', ylabel='Accuracy', title='Validation Accuracy')
        ax.grid()
        fig.savefig("plot_validation_acc.png")
        plt.show()
    if choice==2:
        fig, ax = plt.subplots()
        ax.plot(s, t, color='gold', marker='o',markerfacecolor='darkgoldenrod', markersize=9)
        ax.set(xlabel='alpha- Learning Rate', ylabel='Accuracy', title='Training Accuracy')
        ax.grid()
        fig.savefig("plot_training_acc.png")
        plt.show()
    if choice==3:
        fig, ax = plt.subplots()
        ax.plot(s, t, color='palevioletred', marker='o',markerfacecolor='crimson', markersize=9)
        ax.set(xlabel='alpha- Learning Rate', ylabel='Validation', title='Validation Loss')
        ax.grid()
        fig.savefig("plot_validation_loss.png")
        plt.show()
    if choice==4:
        fig, ax = plt.subplots()
        ax.plot(s, t, color='mediumorchid',  marker='o',markerfacecolor='indigo', markersize=9)
        ax.set(xlabel='alpha- Learning Rate', ylabel='Validation', title='Training Loss')
        ax.grid()
        fig.savefig("plot_training_loss.png")
        plt.show()

def LR():
    # your logistic regression 

    #*********MAIN LOGISTIC REGRESSION*************
    df1=pandas.read_csv('train.csv')
    #df_shape(df1)
    #y_emotions=df1.emotions.values
    #y_shape(y_emotions)
    #print(df1)
    #print(pandas.unique(y_emotions))
    vocabulary=create_word_list(df1)
    dict=create_bag_of_words(df1,vocabulary)
    #print(dict)
    feature_vector=create_feature_vector(df1, vocabulary, dict)
    feature_vector=np.array(feature_vector)
    #print('feature_vector')
    #print_list(feature_vector)
    #print(feature_vector.shape)
    #print('feature vector'+str(feature_vector.shape))
    y_train=create_y_vector(df1)
    #print('y_train'+str(y_train.shape))
    df2=pandas.read_csv('test.csv')
    t_vocabulary=create_word_list(df2)
    #t_dict=create_bag_of_words(df2,t_vocabulary)
    #print(t_dict)
    t_feature_vector=create_feature_vector(df1, t_vocabulary, dict)
    t_feature_vector=np.array(t_feature_vector)
    #print('t_feature_vector')
    #print_list(t_feature_vector)
    #print(t_feature_vector)
    #print(t_feature_vector.shape)
    episodes, trial, alpha=LR_param_init()
    #print(str(episodes)+" "+str(trial)+" "+str(alpha))
    weights=np.random.rand(feature_vector.shape[1], 6)
    answers=lr_function(episodes, trial, alpha, feature_vector, y_train, t_feature_vector, weights)
    format_result(answers, df2)
    #print(df2)
    #print(df1)
    labels=output_labels(df1)
    #print('labels')
    #print(labels)
    #print(labels.shape)


    #*********CROSS VALIDATION LOGISTIC REGRESSION*********
    #episodes=[50, 100, 500]
    #trials=[10, 20, 30]
    '''
    episodes=[600]
    trials=[250]
    regs=[0.001]
    alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0]
    va = []
    ta = []
    vl = []
    tl = []
    for e in episodes:
        for t in trials:
            for a in alpha:
                for r in regs:
                    print(e, t, a, r)
                    k=5 #folds
                    p=0
                    mini=int(feature_vector.shape[0]/k)
                    va2=[]
                    ta2=[]
                    vl2=[]
                    tl2=[]
                    for ak in range(k):
                        lr_cross_validation_function1(feature_vector, y_train, labels, mini, p, e, a, t, r, va2, ta2, vl2, tl2)
                    va.append(np.mean(va2))
                    ta.append(np.mean(ta2))
                    vl.append(np.mean(vl2))
                    tl.append(np.mean(tl2))
    make_plots(alpha,va,1)
    make_plots(alpha,ta, 2)
    make_plots(alpha,vl, 3)
    make_plots(alpha,tl, 4)
    '''

#***********Neural Network***********

#use softmax and find argmax of probabilities
def feed_forward(temp,o1,o2,o3,o4):
    l1=np.dot(temp,o1)+o3
    a=sigmoid(l1)
    l2=np.dot(a,o2)+o4 
    b=softmax(l2)
    return a, b

#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#format results properly in the excel file
def format_result_nn(answers, df2):
    vector_output=pandas.DataFrame(answers, columns=['emotions'])
    leng=len(answers)
    #print(leng)
    #print(len(vector_output['emotions']))
    for i in range(len(vector_output['emotions'])):
        emotion=vector_output['emotions'][i]
        if emotion==0:
            vector_output['emotions'][i]='sadness'
        elif emotion==1:
            vector_output['emotions'][i]='joy'
        elif emotion==2:
            vector_output['emotions'][i]='love'
        elif emotion==3:
            vector_output['emotions'][i]='anger'
        elif emotion==4:
            vector_output['emotions'][i]='fear'
        elif emotion==5:
            vector_output['emotions'][i]='surprise'
    df2['emotions']=vector_output
    df2.to_csv('test_nn.csv',index=False)
    
#loss 
def loss(y,z):
    ans=np.multiply(y,np.log(z))
    return -(np.sum(ans)/1200)

#learned the best parameters using cross validation and feeding those 
def nn_init():
  episodes=1300
  trials=1200
  r=0.001    
  beta=1
  layer1=32
  layer2=6
  return [episodes, trials, beta, layer1, layer2, r]

#creating the nerual network
def create_nn(feature_vector, layer1, layer2):
  o1=np.random.randn(feature_vector.shape[1], layer1) 
  o2=np.random.randn(layer1,layer2) 
  o3=np.random.randn(1,layer1)
  o4=np.random.randn(1,layer2) 
  return [o1, o2, o3, o4]

#separate function for cross validation
def run_nn_crossvalidation(iterations,batch,learn,reg,layer1,feature_vector_80,feature_vector_20,y_train_80,y_train_20,labels_20,labels_80):
    layer2=6 #number of outputs
    #print(features_hidden.shape)
    features_hidden=np.random.randn(feature_vector_80.shape[1], layer1)
    hidden_6=np.random.randn(layer1,layer2) 
    one_hidden=np.random.randn(1,layer1) 
    one_6=np.random.randn(1,layer2)
    for i in range(iterations):
        k=0
        while(k<len(feature_vector_80)):
            a=feature_vector_80[k:min(k+batch,1200)]
            y1=y_train_80[k:min(k+batch,1200)]
            p,q=feed_forward(a,features_hidden,hidden_6,one_hidden,one_6)  
            weight2=np.dot(p.T,q-y1)
            bias2=q-y1
            sig_der=p*(1 - p)
            weight1=np.dot(a.T,sig_der*np.dot(q-y1,hidden_6.T))
            bias1=np.dot(q-y1,hidden_6.T)*sig_der
            features_hidden-=learn*(np.add(np.divide(weight1,a.shape[0]),reg*features_hidden))
            hidden_6-=learn*(np.add(np.divide(weight2,a.shape[0]),reg*hidden_6))
            one_hidden-=learn*(np.add(np.divide(np.sum(bias1,axis=0),a.shape[0]), reg*one_hidden))
            one_6-=learn*(np.add(np.divide(np.sum(bias2,axis=0),a.shape[0]), reg*one_6))
            k+=batch
    p,q=feed_forward(feature_vector_20,features_hidden,hidden_6,one_hidden,one_6)
    ans=np.argmax(q,axis=1)
    p1, q1=feed_forward(feature_vector_80,features_hidden,hidden_6,one_hidden,one_6)
    ans1=np.argmax(q1,axis=1)
    return np.sum(y_train_20 == ans)/len(ans) * 100,np.sum(y_train_80 == ans1)/len(ans1) * 100, loss(labels_20,q), loss(labels_80,q1)

def NN():
    # your Multi-layer Neural Network

    #***************MAIN NEURAL NETWORK******************
    df1=pandas.read_csv('train.csv')
    vocabulary=create_word_list(df1)
    dict=create_bag_of_words(df1,vocabulary)
    #print('dict')
    #print(dict)
    feature_vector=create_feature_vector(df1, vocabulary, dict)
    feature_vector=np.array(feature_vector)
    #print('feature vector'+str(feature_vector.shape))
    #print(feature_vector)
    y_train=create_y_vector(df1)
    #print('y_train'+str(y_train.shape))
    #print(y_train)
    df2=pandas.read_csv('test.csv')
    t_vocabulary=create_word_list(df2)
    #t_dict=create_bag_of_words(df2,t_vocabulary)
    #print(t_dict)
    t_feature_vector=create_feature_vector(df1, t_vocabulary, dict)
    t_feature_vector=np.array(t_feature_vector)
    #print('t_feature_vector'+str(t_feature_vector))
    #print(t_feature_vector[110])
    #arr=feature_vector
    #y=y_train
    #arr_test=t_feature_vector
    #r=0.001
    episodes, trials, beta, layer1, layer2, r =nn_init()
    o1, o2, o3, o4=create_nn(feature_vector, layer1, layer2)
    #p,q=run_nn(episodes, feature_vector, y_train, trials, beta, t_feature_vector, w1, w2, b1, b2)
    for i in range(episodes):
        k=0
        while(k<len(feature_vector)):
            a=feature_vector[k:min(k+trials,1200)]
            y1=y_train[k:min(k+trials,1200)]
            p,q=feed_forward(a,o1,o2,o3,o4)  
            weight2=np.dot(p.T,q-y1)
            bias2=q-y1
            sig_der=p*(1 - p)
            weight1=np.dot(a.T,sig_der * np.dot(q-y1,o2.T))
            bias1=np.dot(q-y1,o2.T) * sig_der
            o1-=beta*(np.add(np.divide(weight1,a.shape[0]), r*o1))
            o2-=beta*(np.add(np.divide(weight2,a.shape[0]), r*o2))
            o3-=beta*(np.add(np.divide(np.sum(bias1,axis=0),a.shape[0]), r*o3))
            o4-=beta*(np.add(np.divide(np.sum(bias2,axis=0),a.shape[0]), r*o4))
            k+=trials
    l1=np.dot(t_feature_vector,o1)+o3
    a=sigmoid(l1)
    l2=np.dot(a,o2)+o4 
    b=softmax(l2)
    answer=np.argmax(b,axis=1)
    format_result_nn(answer, df2)
    #print(answer)
    #df2.to_csv('test_copy.csv',index=False)

    labels=output_labels(df1)

    #*****************CROSS VALIDATION*****************
    '''
    episodes=[600]
    trials=[250]
    regs=[0.001]
    layer=[32]
    alpha=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0]
    va = []
    ta = []
    vl = []
    tl = []
    for e in episodes:
        for t in trials:
            for a in alpha:
                for r in regs:
                    for l in layer:
                        print(e, t, a, r, l)
                        k=5 #folds
                        p=0
                        mini=int(feature_vector.shape[0]/k)
                        va2=[]
                        ta2=[]
                        vl2=[]
                        tl2=[]
                        for ak in range(k):
                            feature_vector_20=feature_vector[p:(p+mini)]
                            feature_vector_80=np.concatenate((feature_vector[:p], feature_vector[(p+mini):]))
                            y_train_20=y_train[p:(p+mini)]
                            y_train_80=np.concatenate((y_train[:p], y_train[(p+mini):]))
                            labels_20=labels[p:(p+mini)]
                            labels_80=np.concatenate((labels[:p], labels[(p+mini):]))
                            va2, ta2, vl2, tl2=run_nn_crossvalidation(e,t,a,r,l, feature_vector_80,feature_vector_20,y_train_80,y_train_20,labels_20,labels_80)
                        va.append(np.mean(va2))
                        ta.append(np.mean(ta2))
                        vl.append(np.mean(vl2))
                        tl.append(np.mean(tl2))
    make_plots(alpha,va,1)
    make_plots(alpha,ta, 2)
    make_plots(alpha,vl, 3)
    make_plots(alpha,tl, 4)'''




if __name__ == '__main__':
    print ("..................Beginning of Logistic Regression................")
    LR()
    print ("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print ("..................Beginning of Neural Network................")
    NN()
    print ("..................End of Neural Network................")