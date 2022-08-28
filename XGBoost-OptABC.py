#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:47:37 2021

@author: Leila Zahedi
"""

import numpy as np
import pandas as pd
import random
#import matplotlib.pyplot as plt
#import csv
#import datetime
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler




#students = pd.read_csv('/Users/leila/Desktop/HPO/ABC/Ready.csv')

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
    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    [(cols[i], OneHotEncoder(), [clmns[i]]) for i in range(len(clmns))],
    remainder='passthrough'   # Leave the rest of the columns untouched
)

X['term_enter']= X['term_enter'].astype(str)

X = ct.fit_transform(X)

all_column_names = list(ct.get_feature_names())

import scipy
if scipy.sparse.issparse(X):
    X=X.todense()


X=pd.DataFrame(X, columns=all_column_names)



# Cross validation or 80:20 train/test sets
#----------------------------------------------------Split Dataset-------------------------------------------------------
y1= y.astype('category')
y = y1.cat.codes

#80:20 sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)
#CV
cv= KFold(n_splits=3, random_state=100, shuffle=True)

#------------------------------------------------
#***Search Space***
#------------------------------------------------

search = { # [low,high)
  'n_estimators': {"type":"int", "range":[5, 500]},
  'learning_rate': {"type":"float", "range":[0,1]},
  'max_depth': {"type":"int", "range":[5,50]},
  "subsample":{"type":"float", "range":[0,1]},
  "colsample_bytree":{"type":"float", "range":[0,1]}
       }

#------------------------------------------------
#***ABC parameters***
#------------------------------------------------
FOOD_NUMBER=5000 #  2000, 5000, 10000
CLUSTERS=int(FOOD_NUMBER/100) #second population number #10,20,50
Limit= 3
iter=20

#------------------------------------------------
#***Initialize variables***
DIMENSION=len(search)
EvalNum=0
MaxEval=2000# FOOD_NUMBER+ iter*(2*FOOD_NUMBER) + iter
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
food_centroid= np.zeros((CLUSTERS))
Min_Acc= 0
Stop_Acc= 0.8864


#------------------------------------------------
#***Update variables according to k-means filtering***
#------------------------------------------------
def update_variables(index_to_drop):
    """
    Updating variables sizes after clustering
    Filtering the food sources with rich solutions changes the size of food
    sources, objective function, fitness and trial matrixes.
    
    Args: the indexs of the food source that is not rich enough
    """
    global CLUSTERS, f, fitness, foods , trial
    CLUSTERS=CLUSTERS-1
    f= np.delete(f, index_to_drop, axis=0) #obj func
    fitness = np.delete(fitness, index_to_drop, axis=0) #fitness
    trial= np.delete(trial, index_to_drop, axis=0) # trial
    foods = np.delete(foods, index_to_drop, axis=0) # foods


#------------------------------------------------
#***filter population based on the defined threshold Min_Acc***
#------------------------------------------------
def filter_pop(Min_Acc):
    """
    Filtering the new population (centroids) with rich food sources 
    
    Args: the minimum accuracy threshold for eliminating food sources
    """
    global CLUSTERS
    temp_f=f 
    #print(temp_f)
    i=CLUSTERS-1
    while i>=0:
        if temp_f[i]<Min_Acc:
           update_variables(i)
        i=i-1
    print("New population after filtering:\n" + str(f)) 
    #print("new foods")
    #print(foods)


#------------------------------------------------
#***Objective Function ***
#------------------------------------------------
def calculate_function(sol):
    """
    Calculate the objective function for each solution in the population 
    
    Args: solution
    
    Returns: Accuracy of the solution
    """
    global EvalNum
    EvalNum=EvalNum+1

    #acc= sol[0]+sol[1]+sol[2]+sol[3]+sol[4] # x+y+z+q+w+s
        
    #Cross Validation
    model= XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                          n_estimators= int(sol[0]), learning_rate=(sol[1]), 
                          max_depth= int(sol[2]),subsample= (sol[3]), 
                          colsample_bytree= (sol[4]), random_state=100, verbosity=0)
    acc=np.mean(cross_val_score(model, X, y, cv=cv, n_jobs=-1,scoring="accuracy"))
    
    #for 80:20
    #model= XGBClassifier(objective='binary:logistic', use_label_encoder=False,
    #                      n_estimators= int(sol[0]), learning_rate=(sol[1]), 
    #                      max_depth= int(sol[2]),subsample= (sol[3]), 
    #                      colsample_bytree= (sol[4]), random_state=100, verbosity=0)
    #model.fit(X_train,y_train)
    #predictions=model.predict(X_test)
    #acc= accuracy_score(y_test, predictions)

    print("F(x):" + str(acc))
    print("Evaluation number: "+ str(EvalNum))
    return (acc),

#------------------------------------------------
#***Fitness Function ***
#------------------------------------------------
def calculate_fitness(fun):
    """
    Calculate the fitness function from objective function
    the formula is : 1/(1+acc)
    
    Args: results from objective function
    
    Returns: fitness
    """
    global EvalNum
    global start_time
    try:
        result = 1 / (fun + 1)
        #print("Fitness:" + str(result))
        print('Duration: {} seconds'.format(time.time() - start_time))
        return result
    except ValueError as err:
        print("An Error occured: " + str(err))
        exit()

#------------------------------------------------
#***Stopping condition***
#------------------------------------------------
def stop_condition():
    """
    Define the conditions where ABC should stop and return the results
    """
    global round
    global EvalNum
    stp = bool(EvalNum >= MaxEval or iter<round or globalOpt>=Stop_Acc)
    return stp

#------------------------------------------------
#***Init food source for scout***
#------------------------------------------------
def init_scout(i):
    """
    Generate two different food sources based on Random and OBL strategies
    
    Args: index of the exhausted food source
    
    Returns: the food source with better quality (acc)
    """
    print("OBL based Food #"+ str(i))
    j=0
    for key in search:
            if search[key]["type"] == "int" or search[key]["type"] == "ctg":
                foods_OBL[0][j]=(search[list(search.keys())[j]]["range"][0]+
                             search[list(search.keys())[j]]["range"][1])-foods[i][j]
                #print(str(key) +": "+str(foods_OBL[0][j]))
            else:
                foods_OBL[0][j]=(search[list(search.keys())[j]]["range"][0]+
                             search[list(search.keys())[j]]["range"][1])-foods[i][j]
                #print(str(key) +": "+str(foods_OBL[0][j]))
            j=j+1
    
    print("Random food #"+ str(i))
    j=0
    for key in search:
            if search[key]["type"] == "int" or search[key]["type"] == "ctg":
                foods_OBL[1][j]=np.random.randint(search[list(search.keys())[j]]["range"][0],
                                           search[list(search.keys())[j]]["range"][1]+1)
                #print(str(key) +": "+str(foods_OBL[1][j]))
    
            else:
                foods_OBL[1][j]=np.round((np.random.uniform(search[list(search.keys())[j]]["range"][0],
                                           search[list(search.keys())[j]]["range"][1])),10)
                #print(str(key) +": "+str(foods_OBL[1][j]))
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
        
    else:
        print(str(second)+" is better than "+  str(first) + "(RANDOM)")
        foods[i]=foods_OBL[1]
        r_food = np.copy(foods[i][:])
        print(str(r_food))
        f[i] = second
        fitness[i] = calculate_fitness(f[i])
        trial[i] = 0
            
#------------------------------------------------
#***clustering initial population***
#------------------------------------------------
def init_pop(i):
    """
    Generate the very first population (food sources)
    For each parameter one random number is generated depending on the type.
    This population will go through kmeans clustering later on
    
    Args: index of the food source
    """
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

#------------------------------------------------
#***clustering initial population***
#------------------------------------------------
def select_cat_centroid(cat_centroids):
    """
    Takes only the category centroilds only (after one hot encoding and kmeans)
    and return the maximum (Final) category for each of the food sources
    
    Args: Only the centroids of categorical parameter (kernel)
    """
    cat_selected = []
    #print("cat_centroids")
    #print(cat_centroids)
    for i in range(cat_centroids.shape[0]):
        row = list(cat_centroids[i,:])
        cat_selected.append(row.index(max(row)))
    return cat_selected
    

#------------------------------------------------
#***K-Means Clustering***
#------------------------------------------------
def init_kmeans(j):
    """
    Operate K-Means clustering in the initial large population
    Applying the one-hot-encoding because we have categorical variable
    and return the final (Max) category for each of the food sources
    
    Args: 
    """
    global foods
    #kproto= KPrototypes(n_clusters=j, verbose=2, max_iter=50)
    #SVM
    # for key in search:
    #     if search[key]["type"] == "ctg":
    #         cat_index=list(search.keys()).index(key)
    #         p_food_cat = pd.get_dummies(p_foods[:,cat_index]) # one-hot for the second HP
    #         p_food_one_hot = np.hstack((np.array([p_foods[:,0]]).T, p_food_cat))# attach first col and dummies as one db
    #         search[key]['cat_num'] = p_food_cat.shape[1]
    #         kmeans = KMeans(n_clusters=j, random_state=0).fit(p_food_one_hot) # cluster the new db
    #         #kproto.fit(p_foods, categorical=1)
    #         cat_centroids = kmeans.cluster_centers_[:,-1*p_food_cat.shape[1]:]# take the centroids and then choose the 2nd HP numbers
    #         cat_selected = select_cat_centroid(cat_centroids)# choose the max value of centroids taken from 2nd HP
    #         new_centroids = np.hstack((np.array([kmeans.cluster_centers_[:,0]]).T, np.array([cat_selected]).T))# attach the 1st col and max from prev step
    #         foods[:j] = new_centroids
    #         print(foods)
    #     else:
    #         kmeans = KMeans(n_clusters=j, random_state=0).fit(p_foods)
    #         centroids = kmeans.cluster_centers_
    #         foods[:j] = centroids
    #         print(foods)
    
    p_foods_tmp = [] # final matrix after k-means
    for key in search:
        _index=list(search.keys()).index(key)
        #print("_index")
        #print(_index)
        if search[key]["type"] == "ctg": #check if the HP is categorical
            p_food_cat = pd.get_dummies(p_foods[:,_index]) # one-hot for the categorical HP
            search[key]['cat_num'] = p_food_cat.shape[1] # take the size of the category
            #print("p_food_cat")
            #print(p_food_cat)
            if len(p_foods_tmp) == 0: # if p_foods_tmp is empty
                p_foods_tmp = p_food_cat
            else: # if p_foods_tmp has already items in it
                if len(p_foods_tmp.shape) == 1: # if the shape is 1 means it's a vector (row-like)
                    p_foods_tmp = np.hstack((np.array([p_foods_tmp]).T, p_food_cat)) #then p_foods_tmp needs transpose
                else:
                    p_foods_tmp = np.hstack((p_foods_tmp, p_food_cat)) #the p_foods_tmp doesnt need transpose
        else: # key is not category
            p_food_num = p_foods[:,_index] #the exact column will be added to a new matrix
            #print("p_food_num")
            #print(p_food_num)
            if len(p_foods_tmp) == 0: # if final food is empty
                p_foods_tmp = p_food_num #assign the p_food_num to the final matrix
            else:  #if final matrix already has items
                if len(p_foods_tmp.shape) == 1: #if final has only one columns then it's a vector
                    #needs transpose, p_food_num needs transpode anyways bcoz it's float and always one column
                    p_foods_tmp = np.hstack((np.array([p_foods_tmp]).T, np.array([p_food_num]).T))
                else: #has more than one columns
                    #no transpose needed , p_food_num needs transpode anyways bcoz it's float and always one column
                    p_foods_tmp = np.hstack((p_foods_tmp, np.array([p_food_num]).T))
    print("\nOne hot encoding Done!")
    #print(p_foods_tmp.shape)
    #print(p_foods_tmp)
    #scaler_cent = StandardScaler()
    scaler_cent = MinMaxScaler()
    p_foods_tmp_s = scaler_cent.fit_transform(p_foods_tmp)
    #Xtest=pd.DataFrame(Xtest, columns=cols_rest)
    print("\nFeature Scaling Done!")
    #print(p_foods_tmp_s)
    #print("inverse")
    #print(scaler_cent.inverse_transform(p_foods_tmp_s))
    #kmeans clustering after one hot encoding
    kmeans = KMeans(n_clusters=j, random_state=0).fit(p_foods_tmp_s)
    print("\nK-Means Clustering Done!")
    #taking the centroids
    clus_centers=kmeans.cluster_centers_
    print("\nCluster centers are taken!")
    #print(clus_centers)
    
    #print("\nclus_centers scale back")
    clus_centers_i=scaler_cent.inverse_transform(clus_centers)
    print("\nCluster centers are back to original scales!")
    #print(np.round(clus_centers_i))
    #raise
    column_ind = 0 # cursor
    new_centroids = []
    #print("kmeans.cluster_centers_")
    #print(kmeans.cluster_centers_)
    #print(kmeans.cluster_centers_.shape)
    for key in search:
        _index=list(search.keys()).index(key)
        if search[key]["type"] == "ctg":
            # take the centroids from where cursor is to the end of all the dummies for that category
            cat_centroids = clus_centers_i[:,column_ind:(column_ind+search[key]['cat_num'])]
            #move the curser to end of that category
            column_ind += search[key]['cat_num']
            #print("search[key]['cat_num']")
            #print(search[key]['cat_num'])
            #print("column_ind")
            #print(column_ind)
            #choose the max value of centroids taken from that category
            cat_selected = select_cat_centroid(cat_centroids)
            if len(new_centroids) == 0: #if final centroids is empty
                new_centroids = np.array([cat_selected]).T #needs trnaspose to be added
            else: #if final centroids has already items in it
                if len(new_centroids.shape) == 1: # if it's size is only 1
                    new_centroids = np.hstack((np.array([new_centroids]).T, np.array([cat_selected]).T))
                else: # if it has more items
                    new_centroids = np.hstack((new_centroids, np.array([cat_selected]).T))
        else: #if not category
            #take the centroid from where cursor is and take that only columns (since not cateogory)
            new_centroids_num = clus_centers_i[:,column_ind]
            #move the cursor only one step (since it's int or float)
            column_ind+=1
            #print("column_ind")
            #print(column_ind)
            if len(new_centroids) == 0: #if final centroids is empty
                new_centroids = new_centroids_num 
            else: #if final centroids has already items in it
                if len(new_centroids.shape) == 1:#if the size is only one => vector => needs transpose
                    new_centroids = np.hstack((np.array([new_centroids]).T, np.array([new_centroids_num]).T))
                else:
                    new_centroids = np.hstack((new_centroids, np.array([new_centroids_num]).T))#no transpose required
                    
    
    print("\nNew centroids are back to non-dummies dataset and ready for training!")
    #print(new_centroids.shape)
    #print(new_centroids)
    #raise
    
    print("\nTaking centroids as the new population and start training!\n")
    for key in search:
        _indexCol=list(search.keys()).index(key)
        if search[key]["type"] == "ctg" or search[key]["type"] == "int":
            new_centroids[:,_indexCol]=np.round(new_centroids[:,_indexCol])
            
    foods[:j] = new_centroids
    #print(foods)
    #print(kmeans.labels_)
    #centroids = kmeans.cluster_centers_

    #labelss=kmeans.labels_
    #centroid_labels = [centroids[i] for i in labelss]
    #print(centroid_labels)

    #foods[:,1]=foods[:,1].round(0).astype(int)

    for i in range(j):
        c_food = np.copy(foods[i][:])
        #print(str(c_food))
        f[i] = calculate_function(c_food)[0]
        fitness[i] = calculate_fitness(f[i])
        trial[i] = 0

#------------------------------------------------
#***Generate all food sources/population***
#------------------------------------------------
start_time = time.time() 

if (not (stop_condition())):
    for k in range(FOOD_NUMBER):
        init_pop(k)
    print("Initial population with the size of " + str(FOOD_NUMBER) + " generated...\n")
    init_kmeans(CLUSTERS)
    filter_pop(Min_Acc)
else:
    print("Stopping condition is already met!")  
      

#Best food source of population
for i in range(CLUSTERS):
        if (f[i] > globalOpt):
            #print(str(f[i]) +">=" + str(globalOpt) + "\t->
            print("\nUpdating optimal solution and parameters...")
            globalOpt = np.copy(f[i])
            globalParams = np.copy(foods[i][:])
print("Best found food source so far: "+ str(globalOpt)+ "\nWith parameters: "+str(globalParams))    
    

while (not(stop_condition())): 
    print("\n\nCycle #"+ str(round))
    
    print("\n\t***Employed Phase***\n")
    i = 0
    while (i < CLUSTERS) and (not(stop_condition())):
        r = random.random()
        print("------------------------")
        print("Employed Bee #"+ str(i)+":")
        param2change = (int)(r * DIMENSION)
        #print("Parameter to change: P" + str(param2change))
        r = random.random()
        neighbour = (int)(r * CLUSTERS)   
        while neighbour == i:
            r = random.random()
            neighbour = (int)(r * CLUSTERS)
        #print("Neighbor to choose: R" + str(neighbour))
        solution = np.copy(foods[i][:])
        #print ("Current Food Source:" + str(solution))
        print ("F(x): " + str(f[i]))
        #print ("Neighbor:" + str(foods[neighbour]))
        r = random.random()
        if search[list(search.keys())[param2change]]["type"]=="float":
            solution[param2change] = foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2
        else:
            solution[param2change] = int(foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2)
        #print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change]) + ", to be replaced with " + str(foods[i][param2change]))    
                #checking the ranges to be whitin accepted values
        if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
                #print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                      #+ " => replace with lower bound") # we may change it later to a new random number
                solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
        if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
                #print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                     #+ " => replace with upper bound")
                solution[param2change] = search[list(search.keys())[param2change]]["range"][1]
        # ---------------------------------------------
        # this is added by me, and is not a step of ABC
        while (solution[param2change]== foods[i][param2change]):
            #print ("New Food Source:" + str(solution))
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
            
               #print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change])+ ", to be replaced with " + str(foods[i][param2change])) 
            
               if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
            
                   #print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                         #+ " => replace with lower bound")
                   solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
        
               else:
                if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
            
                   #print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                         #+ " => replace with upper bound")
                   solution[param2change] = search[list(search.keys())[param2change]]["range"][1]
            
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
        
    if (stop_condition()):
        print("Stopping condition is met!")
    
    print("\n\t***Onlooker Phase***\n")    
    maxfit = np.copy(max(fitness))
    minfit = np.copy(min(fitness))

    prob=[]
    for i in range(CLUSTERS):
        #prob.append(fitness[i] / sum(fitness))
        #prob.append(0.9 *(fitness[i] / maxfit)+0.1)
        prob.append((fitness[i]-minfit)/(maxfit-minfit))

    #print(prob)    
    i = 0
    t = 0
    while (t < CLUSTERS) and (not(stop_condition())):
        r = random.random()
        if (r > prob[i]):
            #print ("Generated random number "+str(r)+" is larger than probability " +str(prob[i])+ " =>\n")
            print("Onlooker Bee #"+ str(t)+" on Food Source #" +str(i))
            t+=1
            param2change = (int)(r * DIMENSION)
            #print("Parameter to change: P" + str(param2change))
            r = random.random()
            neighbour = (int)(r * CLUSTERS)   
            while neighbour == i:
                r = random.random()
                neighbour = (int)(r * CLUSTERS)
            #print("Neighbor to choose: R" + str(neighbour))
            solution = np.copy(foods[i][:])
            #print ("Current Food Source:" + str(solution))
            print ("F(x):" + str(f[i]))
            #print ("Neighbor:" + str(foods[neighbour]))
            r = random.random()
            if search[list(search.keys())[param2change]]["type"]=="float":
                solution[param2change] = foods[i][param2change] + (foods[i][param2change]
                                                               - foods[neighbour][param2change]) * (r - 0.5) * 2
            else:
                solution[param2change] = int(foods[i][param2change] + (foods[i][param2change]
                                                               - foods[neighbour][param2change]) * (r - 0.5) * 2)
            #print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change]) + ", to be replaced with " + str(foods[i][param2change]))
            
                #checking the ranges to be whitin accepted values
            if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
               #print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                     #+ " => replace with lower bound")
               solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
            if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
               #print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                     #+ " => replace with upper bound")
               solution[param2change] = search[list(search.keys())[param2change]]["range"][1]
            # ---------------------------------------------
            # this is added by me, and is not a step of ABC
            while (solution[param2change]== foods[i][param2change]):
                #print ("New Food Source:" + str(solution))
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
                   #print("Choosing another neighbor...")
                   r = random.random()
                   neighbour = (int)(r * CLUSTERS) #choose another neighbor
                   #print ("New neighbor:" + str(foods[neighbour]))
                   r = random.random()
                   if search[list(search.keys())[param2change]]["type"]=="float":
                        solution[param2change] = foods[i][param2change] + (foods[i][param2change]
                                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2
             
                   else:
                        solution[param2change] = int(foods[i][param2change] + (foods[i][param2change]
                                                           - foods[neighbour][param2change]) * (r - 0.5) * 2)
            
                #print("Formula: X[new]=X[i]+ r(X[i]-X[P])= " + str(solution[param2change])+ ", to be replaced with " + str(foods[i][param2change])) 
            
                if solution[param2change] < search[list(search.keys())[param2change]]["range"][0]:
                    #print(str(solution[param2change]) +"<" + str(search[list(search.keys())[param2change]]["range"][0])
                          #+ " => replace with lower bound")
                    solution[param2change] = search[list(search.keys())[param2change]]["range"][0]
                if solution[param2change] > search[list(search.keys())[param2change]]["range"][1]:
                    #print(str(solution[param2change]) +">" + str(search[list(search.keys())[param2change]]["range"][1])
                          #+ " => replace with upper bound")
                    solution[param2change] = search[list(search.keys())[param2change]]["range"][1] 
            
            #print ("Final updated Food Source:" + str(solution))
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
            #print ("r="+str(r)+" is smaller than " +str(prob[i]))
            print ("Onlooker bee goes to the next food source")

        i += 1
        i = i % (CLUSTERS)
        print("------------------------")
    #prob.clear()
    if (stop_condition()):
        print("Stopping condition is met!")

    
    print("\n***Best Result So Far***")
    print("\nUpdating optimal solution and parameters...")
    for i in range(CLUSTERS):
            if (f[i] > globalOpt):
                #print(str(f[i]) +">" + str(globalOpt) + "\t-> 
                globalOpt = np.copy(f[i])
                globalParams = np.copy(foods[i][:])
    print("Best food source so far: "+ str(globalOpt)+ "\nWith parameters: " +str(globalParams))    
          
    
    print("\n***Scout Phase OBL***")
    if (np.amax(trial) >= Limit):
           #print("trial" + str(trial))
           #print("Max Trial >= Limit, occurs at row " + str(trial.argmax(axis = 0)))
           print("Scout explores a random food source...")
           init_scout(trial.argmax(axis = 0))
           if f[trial.argmax(axis = 0)]> globalOpt:
                globalOpt = np.copy(f[trial.argmax(axis = 0)])
                globalParams = np.copy(foods[trial.argmax(axis = 0)][:])
    else:
        print ("Trials < Limit \n=> No scouts are required!")    
    round=round+1

    
#end_time = datetime.datetime.now() #end time
print("------------------------------------------------")
print("\t***Results***")
print("------------------------------------------------")
globalOpts.append(globalOpt)
print("Global Optimum: " + str(max(globalOpts)))
print("Global Parameters: " + str(globalParams))

#duration= format(end_time-start_time)
print('Duration: {} seconds'.format(time.time() - start_time))
print("Number of evaluations:" +str(EvalNum))
print("Found optimal after "+ str(round-1) + " rounds!")

