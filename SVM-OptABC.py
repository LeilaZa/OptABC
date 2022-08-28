#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:01:19 2021

@author: leila
"""

import numpy as np
import pandas as pd
import random
#import matplotlib.pyplot as plt
#import csv
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score #,classification_report,confusion_matrix 
#from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.svm import SVC #,SVR
from sklearn.cluster import KMeans
#from kmodes.kprototypes import KPrototypes



#students = pd.read_csv(r'C:\Users\lzahe001\Desktop\Ready.csv')

students = pd.read_csv('Ready.csv')

X=students.drop('graduated',axis=1)
y=students['graduated']

cols =list(X.select_dtypes(include=['object']).columns)
cols_rest=list(X.select_dtypes(exclude=['object']).columns)
test0=students[cols]
test1=students[cols_rest]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xtest = sc_X.fit_transform(test1)
Xtest=pd.DataFrame(Xtest, columns=cols_rest)

X = pd.concat([Xtest.reset_index(drop=True), test0], axis=1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

clmns=[X.columns.get_loc(c) for c in cols if c in X]

ct = ColumnTransformer(
    [(cols[i], OneHotEncoder(), [clmns[i]]) for i in range(len(clmns))],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X['term_enter']= X['term_enter'].astype(str)

X = ct.fit_transform(X)

all_column_names = list(ct.get_feature_names())

import scipy
if scipy.sparse.issparse(X):
    X=X.todense()


X=pd.DataFrame(X, columns=all_column_names)



# training - set - not CV
#----------------------------------------------------Split Dataset-------------------------------------------------------
y1= y.astype('category')
y = y1.cat.codes
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True,random_state=100)

#------------------------------------------------
#***Search Space***
#------------------------------------------------

search = {
    'C': {"type":"float", "range":[0.1, 50]},
    'kernel':{"type":"ctg", "range":[0,3]},
}
#------------------------------------------------
#***ABC parameters***
#------------------------------------------------
#DIMENSION=3
FOOD_NUMBER=5000 # 1000, 2000, 5000
CLUSTERS=int(FOOD_NUMBER/100) #second population number #10,20,50,100
Limit= 3
iter=5
#------------------------------------------------
#***Initialize variables***
#------------------------------------------------
DIMENSION=len(search)
EvalNum=0
MaxEval=400# FOOD_NUMBER+ iter*(2*FOOD_NUMBER) + iter
#RUN_TIME=2
solution = np.zeros(DIMENSION)
f = np.ones(CLUSTERS)
fitness = np.ones(CLUSTERS) * np.iinfo(int).max
trial = np.zeros(CLUSTERS)
globalOpt = 0
globalParams = [0 for x in range(DIMENSION)]
globalOpts=list()
round = 1
p_foods = np.zeros((FOOD_NUMBER, DIMENSION))
foods = np.zeros((CLUSTERS, DIMENSION))
foods_OBL = np.zeros((2, DIMENSION))
lbl = np.zeros((CLUSTERS))
food_centroid= np.zeros((CLUSTERS))

cv = KFold(n_splits=3, random_state=100, shuffle=True)

#------------------------------------------------
#***Start Timer***
#------------------------------------------------
start_time = datetime.datetime.now()

#------------------------------------------------
#***Objective Function ***
#------------------------------------------------
def calculate_function(sol):
    global EvalNum
    EvalNum=EvalNum+1
    #res= sol[0]+sol[1] # x+y
    if int(sol[1]==0):
        krnl='linear'
    elif int(sol[1]==1) :
        krnl='poly'
    elif int(sol[1]==2) :
        krnl='rbf'
    elif int(sol[1]==3) :
        krnl='sigmoid'
        
    #kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    model = SVC(C= (sol[0]), kernel= krnl, random_state=100)
    #res = cross_val_score(model, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=3)).mean()
    res=np.mean(cross_val_score(model, X, y, cv=cv, n_jobs=-1,scoring="accuracy"))
    #model.fit(X_train,y_train)
    #predictions=model.predict(X_test)
    #res= accuracy_score(y_test, predictions)
    print("F(x):" + str(res))
    print("Evaluation number: "+ str(EvalNum))
    return (res),

#------------------------------------------------
#***Fitness Function ***
#------------------------------------------------
def calculate_fitness(fun):
    global EvalNum
    global start_time
    try:
        result = 1 / (fun + 1)
        print("Fitness:" + str(result))
        timing=  datetime.datetime.now()- start_time
        print("Time:" + str(timing))
        return result
    except ValueError as err:
        print("An Error occured: " + str(err))
        exit()

#------------------------------------------------
#***Stopping condition***
#------------------------------------------------
def stop_condition():
    global round
    global EvalNum
    stp = bool(EvalNum >= MaxEval or iter<round or globalOpt>=0.88) # f= x+y
    return stp

#------------------------------------------------
#***Init food source for scout***
#------------------------------------------------
def init_scout(i): #creating random food source for each row based on OBL
        print("OBL based Food #"+ str(i))
        j=0
        for key in search:
                if search[key]["type"] == "int" or search[key]["type"] == "ctg":
                    foods_OBL[0][j]=(search[list(search.keys())[j]]["range"][0]+
                                 search[list(search.keys())[j]]["range"][1])-foods[i][j]
                    print(str(key) +": "+str(foods_OBL[0][j]))
                else:
                    foods_OBL[0][j]=(search[list(search.keys())[j]]["range"][0]+
                                 search[list(search.keys())[j]]["range"][1])-foods[i][j]
                    print(str(key) +": "+str(foods_OBL[0][j]))
                j=j+1
        
        print("Random food #"+ str(i))
        j=0
        for key in search:
                if search[key]["type"] == "int" or search[key]["type"] == "ctg":
                    foods_OBL[1][j]=np.random.randint(search[list(search.keys())[j]]["range"][0],
                                               search[list(search.keys())[j]]["range"][1]+1)
                    print(str(key) +": "+str(foods_OBL[1][j]))
        
                else:
                    foods_OBL[1][j]=np.round((np.random.uniform(search[list(search.keys())[j]]["range"][0],
                                               search[list(search.keys())[j]]["range"][1])),10)
                    print(str(key) +": "+str(foods_OBL[1][j]))
                j=j+1
                
        first= calculate_function(foods_OBL[0])[0]
        second= calculate_function(foods_OBL[1])[0]
        if first>second:
            print(str(first)+" is better than "+  str(second) +"(OBL)")
            foods[i]=foods_OBL[0]
            r_food = np.copy(foods[i][:])
            print(str(r_food))
            f[i] = first
            fitness[i] = calculate_fitness(f[i])
            trial[i] = 0
            print("------------------------")
            
        else:
            print(str(second)+" is better than "+  str(first) + "(RANDOM)")
            foods[i]=foods_OBL[1]
            r_food = np.copy(foods[i][:])
            print(str(r_food))
            f[i] = second
            fitness[i] = calculate_fitness(f[i])
            trial[i] = 0
            print("------------------------")
            

        #print("------------------------")

#------------------------------------------------
#***clustering initial population***
#------------------------------------------------
def init_pop(i): #creating population
    j=0
    for key in search:
        if search[key]["type"] == "int" or search[key]["type"] == "ctg":
            p_foods[i][j]=np.random.randint(search[list(search.keys())[j]]["range"][0],
                                               search[list(search.keys())[j]]["range"][1]+1)
            #print(str(key) +": "+str(p_foods[i][j]))
        
        else:
            p_foods[i][j]=np.round((np.random.uniform(search[list(search.keys())[j]]["range"][0],
                                               search[list(search.keys())[j]]["range"][1])),10)
            #print(str(key) +": "+str(p_foods[i][j]))
        j=j+1


def select_cat_centroid(cat_centroids):
    cat_selected = []
    for i in range(cat_centroids.shape[0]):
        row = list(cat_centroids[i,:])
        cat_selected.append(row.index(max(row)))
    return cat_selected
    


def init_kmeans(j): #creating population with k-means
    global foods
    #kproto= KPrototypes(n_clusters=j, verbose=2, max_iter=50)
    p_food_cat = pd.get_dummies(p_foods[:,1])
    p_food_one_hot = np.hstack((np.array([p_foods[:,0]]).T, p_food_cat))
    kmeans = KMeans(n_clusters=j, random_state=0).fit(p_food_one_hot) 
    #kproto.fit(p_foods, categorical=1)
    cat_centroids = kmeans.cluster_centers_[:,-1*p_food_cat.shape[1]:]
    cat_selected = select_cat_centroid(cat_centroids)
    new_centroids = np.hstack((np.array([kmeans.cluster_centers_[:,0]]).T, np.array([cat_selected]).T))
    foods[:j] = new_centroids
    print(kmeans.labels_)
    centroids = kmeans.cluster_centers_

    #labelss=kmeans.labels_
    #centroid_labels = [centroids[i] for i in labelss]
    #print(centroid_labels)

    print("K-Means Clustering is done!")
    foods[:,1]=foods[:,1].round(0).astype(int)
    print("Taking centroids as part of the new population...")
    print(foods)
    #print("------------------------------------------------")

    for i in range(j):
        c_food = np.copy(foods[i][:])
        print(str(c_food))
        f[i] = calculate_function(c_food)[0]
        fitness[i] = calculate_fitness(f[i])
        trial[i] = 0
        #print("------------------------------------------------")


#------------------------------------------------
#***Generate all food sources/population***
#------------------------------------------------
print("------------------------------------------------")
print("\t***Population***")
print("------------------------------------------------")
if (not (stop_condition())):
    for k in range(FOOD_NUMBER):
        init_pop(k)
        #print("------------------------------------------------")
    print("Initial population with the size of " + str(FOOD_NUMBER) + " generated...")
    print("------------------------------------------------")
    init_kmeans(CLUSTERS)
    print("------------------------------------------------")

else:
    print("Stopping condition is already met!")  
      

print("------------------------------------------------")
print("\t***Best food source of population***")
print("------------------------------------------------")
for i in range(CLUSTERS):
        if (f[i] >= globalOpt):
            print(str(f[i]) +">=" + str(globalOpt) + "\t->Updating optimal solution and parameters...\n")
            globalOpt = np.copy(f[i])
            globalParams = np.copy(foods[i][:])
print("\nBest found food source in this round: "+ str(globalOpt)+ "\nWith parameters: "+str(globalParams))    
    

while (not(stop_condition())): 
    print("------------------------------------------------------------------------------------------------")
    print("\t\t\tCycle #"+ str(round))
    
    print("------------------------------------------------")
    print("\t***Employed Phase***")
    print("------------------------------------------------")

    i = 0
    while (i < CLUSTERS) and (not(stop_condition())):
        r = random.random()
        print("Employed Bee "+ str(i)+":")
        print("------------------------")
        param2change = (int)(r * DIMENSION)
        print("Parameter to change: P" + str(param2change))
        r = random.random()
        neighbour = (int)(r * CLUSTERS)   
        while neighbour == i:
            r = random.random()
            neighbour = (int)(r * CLUSTERS)
        print("Neighbor to choose: R" + str(neighbour))
        solution = np.copy(foods[i][:])
        print ("Current Food Source:" + str(solution))
        print ("F(x): " + str(f[i]))
        print ("Neighbor:" + str(foods[neighbour]))
        r = random.random()
        if search[list(search.keys())[param2change]]["type"]=="float":
            solution[param2change] = foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2
        else:
            solution[param2change] = int(foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2)
        print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change]) + ", to be replaced with " + str(foods[i][param2change]))    
                #checking the ranges to be whitin accepted values
        if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
                print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                      + " => replace with lower bound") # we may change it later to a new random number
                solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
        if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
                print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                     + " => replace with upper bound")
                solution[param2change] = search[list(search.keys())[param2change]]["range"][1]
        # ---------------------------------------------
        # this is added by me, and is not a step of ABC
        while (solution[param2change]== foods[i][param2change]):
            print ("New Food Source:" + str(solution))
            print("Current food source and new food source are the same. trying again...")
            if list(search.keys())[param2change]=="criterion":
               print("Flipping criterion value")
               solution[param2change] = int(foods[i][param2change])^1
            # elif list(search.keys())[param2change]=="max_features":
            #    print("changing max_features value")
            #    r = [*range(search[list(search.keys())[param2change]]["range"][0],foods[i][param2change]),
            #         *range(foods[i][param2change]+1,search[list(search.keys())[param2change]]["range"][0]+1)]
            #    solution[param2change]=random.choice(r)
            else:
               r = random.random()
               neighbour = (int)(r * CLUSTERS) #choose another neighbor
               r = random.random()
               if search[list(search.keys())[param2change]]["type"]=="float":
                        solution[param2change] = foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2
               else:
                        solution[param2change] = int(foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2)
            
               print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change])+ ", to be replaced with " + str(foods[i][param2change])) 
            
               if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
            
                   print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                         + " => replace with lower bound")
                   solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
        
               else:
                if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
            
                   print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                         + " => replace with upper bound")
                   solution[param2change] = search[list(search.keys())[param2change]]["range"][1]
        # ---------------------------------------------
            
        print ("Updated Food Source:" + str(solution))
        ObjValSol = calculate_function(solution)[0]
        FitnessSol = calculate_fitness(ObjValSol)

        
        #Replace the results if better and reset trial    
        if  (FitnessSol <= fitness[i]):
                print("The solution improved! Updating the results & resetting trial.... ")
                trial[i] = 0
                foods[i][:] = np.copy(solution)
                f[i] = ObjValSol
                fitness[i] = FitnessSol
        else:
                print("The solution didn't improve! Incrementing trial.... ")
                trial[i] = trial[i] + 1
        i += 1
        print("------------------------")
        
    if (stop_condition()):
        print("Stopping condition is met!")
    
    
    print("------------------------------------------------")
    print("\t***Onlooker Phase***")
    print("------------------------------------------------")
    
    maxfit = np.copy(max(fitness))
    prob=[]
    for i in range(CLUSTERS):
        prob.append(fitness[i] / maxfit)
    i = 0
    t = 0
    while (t < CLUSTERS) and (not(stop_condition())):
        r = random.random()
        if (r > prob[i]):
            print ("Generated random number "+str(r)+" is larger than probability " +str(prob[i])+ " =>\n")
            print("Onlooker Bee "+ str(t)+":")
            print("On food source "+ str(i))

            t+=1
            print("------------------------")
            param2change = (int)(r * DIMENSION)
            print("Parameter to change: P" + str(param2change))
            r = random.random()
            neighbour = (int)(r * CLUSTERS)   
            while neighbour == i:
                r = random.random()
                neighbour = (int)(r * CLUSTERS)
            print("Neighbor to choose: R" + str(neighbour))
            solution = np.copy(foods[i][:])
            print ("Current Food Source:" + str(solution))
            print ("F(x):" + str(f[i]))
            print ("Neighbor:" + str(foods[neighbour]))
            r = random.random()
            if search[list(search.keys())[param2change]]["type"]=="float":
                solution[param2change] = foods[i][param2change] + (foods[i][param2change]
                                                               - foods[neighbour][param2change]) * (r - 0.5) * 2
            else:
                solution[param2change] = int(foods[i][param2change] + (foods[i][param2change]
                                                               - foods[neighbour][param2change]) * (r - 0.5) * 2)
            print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change]) + ", to be replaced with " + str(foods[i][param2change]))
            
                #checking the ranges to be whitin accepted values
            if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
               print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                     + " => replace with lower bound")
               solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
            if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
               print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                     + " => replace with upper bound")
               solution[param2change] = search[list(search.keys())[param2change]]["range"][1]
            # ---------------------------------------------
            # this is added by me, and is not a step of ABC
            while (solution[param2change]== foods[i][param2change]):
                print ("New Food Source:" + str(solution))
                print("Current food source and new food source are the same. trying again...")
                if list(search.keys())[param2change]=="criterion":
                   print("Flipping criterion value...")
                   solution[param2change] = int(foods[i][param2change])^1
                # elif list(search.keys())[param2change]=="max_features":
                #    print("changing max_features value")
                #    r = [*range(search[list(search.keys())[param2change]]["range"][0],foods[i][param2change]),
                #         *range(foods[i][param2change]+1,search[list(search.keys())[param2change]]["range"][0]+1)]
                #    solution[param2change]=random.choice(r)
                else:
                   print("Choosing another neighbor...")
                   r = random.random()
                   neighbour = (int)(r * CLUSTERS) #choose another neighbor
                   print ("New neighbor:" + str(foods[neighbour]))
                   r = random.random()
                   if search[list(search.keys())[param2change]]["type"]=="float":
                        solution[param2change] = foods[i][param2change] + (foods[i][param2change]
                                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2
             
                   else:
                        solution[param2change] = int(foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2)
            
                print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change])+ ", to be replaced with " + str(foods[i][param2change])) 
            
                if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
                    print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                          + " => replace with lower bound")
                    solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
                if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
                    print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                          + " => replace with upper bound")
                    solution[param2change] = search[list(search.keys())[param2change]]["range"][1] 
            
            print ("Final updated Food Source:" + str(solution))
            ObjValSol = calculate_function(solution)[0]
            FitnessSol = calculate_fitness(ObjValSol)

           #replace the results if better
            if  (FitnessSol <= fitness[i]):
                print("The solution improved! Updating the results & resetting trial.... ")
                trial[i] = 0
                foods[i][:] = np.copy(solution)
                f[i] = ObjValSol
                fitness[i] = FitnessSol
            else:
                print("The solution didn't improve! Incrementing trial by one.... ")
                trial[i] = trial[i] + 1  
                
        else:
            print ("r="+str(r)+" is smaller than " +str(prob[i]))
            print ("Onlooker bee goes to the next food source")

        i += 1
        i = i % (CLUSTERS)
        print("------------------------")
        
    if (stop_condition()):
        print("Stopping condition is met!")

    
    print("------------------------------------------------")
    print("\t***Best food source***")
    print("------------------------------------------------")
    
    for i in range(CLUSTERS):
            if (f[i] > globalOpt):
                print(str(f[i]) +">" + str(globalOpt) + "\t-> Updating optimal solution and parameters...\n")
                globalOpt = np.copy(f[i])
                globalParams = np.copy(foods[i][:])
    print("\nBest food source so far: "+ str(globalOpt)+ "\nWith parameters: " +str(globalParams))    
          
    
    print("------------------------------------------------")
    print("\t***Scout Phase OBL***")
    print("------------------------------------------------")
    if (np.amax(trial) >= Limit):
           print("trial" + str(trial))
           print("Max Trial >= Limit, occurs at row " + str(trial.argmax(axis = 0)))
           print("Scout explores a random food source...")
           init_scout(trial.argmax(axis = 0))
           if f[trial.argmax(axis = 0)]> globalOpt:
                globalOpt = np.copy(f[trial.argmax(axis = 0)])
                globalParams = np.copy(foods[trial.argmax(axis = 0)][:])
    else:
        print ("Trials < Limit \n\n=> No scouts are required!")    
    round=round+1

    
end_time = datetime.datetime.now() #end time
print("------------------------------------------------")
print("\t***Results***")
print("------------------------------------------------")
globalOpts.append(globalOpt)
print("Global Optimum: " + str(max(globalOpts)))
print("Global Parameters: " + str(globalParams))

duration= end_time-start_time
print("Tuning time: "+ str(duration))
print("Number of evaluations:" +str(EvalNum))
print("Found optimal after "+ str(round-1) + " rounds!")
