#import Dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import scipy as sc
from scipy import linalg, stats, special
import copy
import pickle
import random
from random import sample
import math

#Initialising all the Models and Data
embed=load_model('embed.h5')
GAN = load_model('CGAN.h5')
file1 = open('mean', 'rb')
file2 = open('covariance','rb')
file3 = open('feedback','rb')
og_mean = pickle.load(file1)
og_cov = pickle.load(file2)
feedback = pickle.load(file3)
size=len(og_mean)
with open('data_sorted.npy', 'rb') as f:
    data = np.load(f)
file1.close()
file2.close()
file3.close()

#Uploading the previous dataset to generate the total mean and covariance
def prediction(dat):
    tmp=[]
    for i in range(len(dat)):
        temp=dat[i]
        temp=np.reshape(temp,(-1,28,28,1))
        tmp.append(embed.predict(x=temp))
    return(np.array(tmp))

label_sort=prediction(data)
x2=label_sort.reshape((98000,2))
#Retrieving the covariance and mean
c=np.cov(x2, rowvar=False)
a,b=sum(x2[:,0])/len(x2),sum(x2[:,1])/len(x2)
m=np.zeros(2)
m[0]=a
m[1]=b
mean=[]
cov=[]
for i in range(size):
    mean.append(m)
    cov.append(c)


class Work:
    def __init__ (self,size,GAN,mean,cov,feedback,og_mean,og_cov,max_iter=100):
        self.size=copy.deepcopy(size)
        self.data=[[] for x in range(size)]# Stores each image for each cluster
        self.X=[] #Stores just the images
        self.GAN=GAN
        self.mean=copy.deepcopy(mean)
        self.og_mean=copy.deepcopy(og_mean)
        self.cov=copy.deepcopy(cov)
        self.og_cov=copy.deepcopy(og_cov)
        self.feedback=copy.deepcopy(feedback)
        self.max_iter=max_iter
        self.decay=0.9

    def initialise(self):
        '''Initialise the label array that will be used to find the user choices
           Initialise the value array. It will help decide the liked and disliked cluster
           Randomly initialise liked and disliked cluster to start
           Creating two arrays to store liked and disliked images
        '''
        self.liked=[]
        self.disliked=[]
        self.label=np.zeros(self.size)

        
    """In the check function, since we are using synthetic data, we will just use the feedback array to find whether the user 
       likes the image or not. In real world testing the facial data using FER algorithm and the decision the user takes will 
       provide the feedback."""    
        
    def check(self,point,cl): #This function mimics whether the user likes the image or not
        if self.feedback[cl]==1:
            return 1
        else:
            return 0

    def E_M_Step(self,feed):
        ''' --------------------------   E - STEP   -------------------------- '''
        # Initiating the r matrix, evrey row contains the probabilities
        # for every cluster for this row
        self.r = np.zeros((len(self.X), self.size))
        self.w=np.random.random(self.size)
        self.w=self.w/np.sum(self.w)
        # Calculating the r matrix
        for i in range(len(self.X)):
            self.r_data = np.zeros(self.size)
            for j in range(self.size):
                try:
                    pf=sc.stats.multivariate_normal.pdf(self.X[i], self.mean[j], self.cov[j])
                    pf=pf+0.00000000000001
                except:
                    print("Error Occured")
                    pf=0.00000000000001
                self.r_data[j] = self.w[j] * pf
            if i==(len(self.X)-1):#Special Updation for the new image
                for j in range(self.size):
                    if feed==1:# For Liked Images
                        if self.label[j]==0:
                            self.r_data[j]=0
                    elif feed==0:# For Disliked Images
                        if self.label[j]==1:
                            self.r_data[j]=0
            for j in range(self.size):
                if np.sum(self.r_data)==0:
                    self.r[i][j]=0.00000000000001
                else:
                    self.r[i][j] = self.r_data[j] / np.sum(self.r_data)
        # Calculating the N
        self.N = np.sum(self.r, axis=0)
        
        '''--------------------------   M - STEP   -------------------------- '''
        # Initializing the mean vector as a zero vector
        self.mean_vector=np.zeros((self.size,2))
        self.covariance=np.zeros((self.size,2,2))
            
        # Updating the mean vector
        for j in range(self.size):
            if (len(self.data[j])>=2): #Updates Normally when when more than one image is in a cluster
                tmp=0
                for i in range(len(self.X)):
                    tmp+=(self.r[i][j]*self.X[i])
                self.mean_vector[j]=(1/self.N[j])*(tmp)
            elif (len(self.data[j])==1): #Stores the image embedding as mean if only one image available in cluster
                image=self.data[j]
                print(image[0])
                self.mean_vector[j]=np.array(image)
            else: #Stores the initialised mean when no image available in cluster
                self.mean_vector[j]=self.mean[j]
            
        #Updating the Covariance
        for j in range(self.size):
            if (len(self.data[j])>=2): #Updates Normally when when more than one image is in a cluster
                tmp=0
                for i in range(len(self.X)):
                    t=self.X[i]-self.mean_vector[j]
                    t1=np.outer(t,t)
                    tmp+=(self.r[i][j]*t1)
                self.covariance[j]=((1/self.N[j])*(tmp))
            else :
                #Stores the initialised mean when no image available in cluster
                self.covariance[j]=self.cov[j]
                
            
        # Updating the weight list
        self.w = [self.N[k]/len(self.X) for k in range(self.size)]
        self.mean=copy.deepcopy(self.mean_vector)
        self.cov=copy.deepcopy(self.covariance)
                            
    def update_labels(self):
        """We will be updating labels here based. We will find the pdf of liked image for each label add them.
        Find the pdf of disliked image for each label and add them. then use it to find the likelihood of the
        label being liked disliked or unknown.
        """
        for q in range(self.size):
            li=0.00000000000001
            di=0.00000000000001
            for img in self.liked:#
                try:
                    pdf=sc.stats.multivariate_normal.pdf(img, self.mean[q], self.cov[q])
                    li=li+pdf
                except:
                    print("Error Occured")
                    print("Image is ", img)
                    print("Mean is ",self.mean[i])
                    print("Covariance is ", self.cov[i])
                    print("Update Iteration is ",i)
                    break
            for img in self.disliked:
                try:
                    pdf=sc.stats.multivariate_normal.pdf(img, self.mean[q], self.cov[q])
                    di=di+pdf
                except:
                    print("Error Occured")
                    print("Image is ", img)
                    print("Mean is ",self.mean[i])
                    print("Covariance is ", self.cov[i])
                    print("Update Iteration is ",i) 
                    
            val=(li-di)/(li+di)
            if val>0:#If value greater than 0, then liked
                self.label[q]=1
            elif val<0:#If value is lesser than 0 then disliked
                self.label[q]=0
            else:#If value is 0, then it is unlabled.
                self.label[q]=2
        print("labels",self.label)
        
    def check_precision(self):
        t_li=[]
        t_di=[]
        tp=0
        fp=0
        tn=0
        fn=0
        
        #Generate Liked and Disliked images (100 for each cluster) based on the predicted label
        for d in range(len(self.mean)):
            if self.label[d]==1:
                for e in range(100):
                    t_li.append((np.random.multivariate_normal(self.mean[d],self.cov[d])))
            if self.label[d]==0:
                for e in range(100):
                    t_di.append((np.random.multivariate_normal(self.mean[d],self.cov[d])))
        
        #Find the likelihood of the image in the clusters and find the cluster with highest probability
        for d in range(len(t_li)):
            pdf=[]
            for e in range(len(self.og_mean)):
                pdf.append(sc.stats.multivariate_normal.pdf(t_li[d], self.og_mean[e], self.og_cov[e]))
            index=np.argmax(pdf)
            if self.feedback[index]==1:
                tp=tp+1
            else:
                fp=fp+1
        for d in range(len(t_di)):
            pdf=[]
            for e in range(len(self.og_mean)):
                pdf.append(sc.stats.multivariate_normal.pdf(t_di[d], self.og_mean[e], self.og_cov[e]))
            index=np.argmax(pdf)
            if self.feedback[index]==0:
                tn=tn+1
            else:
                fn=fn+1
        if tp==0:
            return 0,0,0
        else:
            return (tp/(tp+fp)),(tp/(tp+fn)),((tp+tn)/(tp+tn+fp+fn))
    

    def simulate(self):
        #Call the Initialise function
        self.initialise()
        #Run the simulation
        for i in range(self.max_iter):
            #Generate a random number which will decide whether to explore or exploit.
            print("Iteration",i)
            exp=random.random()
            cl=0
            if (exp<self.decay): #Explore
                #Randomly generate the cluster number
                cl=random.randint(0,(self.size-1))
            else:#Exploit
                '''In the Exploiting Part, We will be randomly choose a cluster which is liked by the User. And update the
                values according to whether the user liked the image or not'''
                li=[]
                for j in range(self.size):
                    if self.label[j]==1:
                        li.append(j)
                if (len(li))==0:#If there are no liked labels
                    cl=random.randint(0,(self.size-1))
                else:
                    cl=random.choice(li)#Choose a Liked cluster    
            #Extract the mean and cov for the cluster
            m=self.mean[cl]
            c=self.cov[cl]
            #Generate a point from the cluster
            point= np.random.multivariate_normal(m,c)
            #print("point ", point)
            feed=self.check(point,cl)
            
            #Storing the data in a two lists
            self.data[cl].append(point)
            self.X.append(point)
            """print(self.data[cl])
            print("\n\n\n")
            print(self.data)"""

            if feed==1: #User liked the image
                self.liked.append(point)
            elif feed==0:#User disliked the video
                self.disliked.append(point)
                
            #Updating the Clusters Using the E-M Step
            self.E_M_Step(feed)
            
            #Updating the Labels depending on the User likeness.
            self.update_labels()
            
            # Decaying the Delta value to create equilibrium between Exploration and Exploitation.
            if i%20==0:
                self.decay=self.decay-0.2


            #Checking if the Agent has learnt the user likeness.
            prec,rec, acc=self.check_precision()
            print("precision",prec)
            print("recall",rec)
            print("accuracy",acc)
            if prec>=0.80 and rec>=0.5 and acc>=0.50:
                print("Good Precision and Recall")
                print("Iteration Value",i)
                break
        ret_val=[]
        ret_val.append(self.label)
        ret_val.append(i)
        ret_val.append(prec)
        ret_val.append(rec)
        ret_val.append(acc)
        ret_val.append(self.liked)
        ret_val.append(self.disliked)
        ret_val.append(self.mean)
        ret_val.append(self.cov)
        ret_val.append(self.X)
        return ret_val
    
for j in range(0,26):
    print("User",j)
    epoch=Work(size,GAN,mean,cov,feedback[j],og_mean,og_cov)
    package=epoch.simulate()
    f1= open('User'+str(j), 'ab')
    pickle.dump(package,f1)
    f1.close()