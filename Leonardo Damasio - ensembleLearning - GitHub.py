print("""\

Ensemble Learning
Objetivo: Detecção de fraude

Autor: Leonardo Damasio | Senior Data Scientist
Data: 19/12/2019

                                                                                                
FFFFFFF RRRRRR    AAA   UU   UU DDDDD      DDDDD   EEEEEEE TTTTTTT EEEEEEE  CCCCC  TTTTTTT IIIII  OOOOO  NN   NN 
FF      RR   RR  AAAAA  UU   UU DD  DD     DD  DD  EE        TTT   EE      CC    C   TTT    III  OO   OO NNN  NN 
FFFF    RRRRRR  AA   AA UU   UU DD   DD    DD   DD EEEEE     TTT   EEEEE   CC        TTT    III  OO   OO NN N NN 
FF      RR  RR  AAAAAAA UU   UU DD   DD    DD   DD EE        TTT   EE      CC    C   TTT    III  OO   OO NN  NNN 
FF      RR   RR AA   AA  UUUUU  DDDDDD     DDDDDD  EEEEEEE   TTT   EEEEEEE  CCCCC    TTT   IIIII  OOOO0  NN   NN 
                                                                                                                 
                                                                                             by Leonardo Damasio



>>> Ensemble Learning

What is an ensemble method?

"Ensemble models in machine learning combine the decisions from multiple models to improve the overall performance." 

                                                                                    Reference: Towards Data Science



>>> Models Used

Logistic Regression
Naive Bayes Classifier
K-Nearest Neighbors Classifier
Decision Tree Classifier
Support Vector Machine Classifier
Random Forest Classifier
Extreme Gradient Boosted Trees Classifier
Deep Learning Multilayer Perceptron Neural Networks



>>> Ensemble Technique

Model Score = m
Accuracy = a
Final Score = Weighted Average = ((m1 * a1 + m2 * a2 + ... + mn * an) / n) / ((a1 + a2 + ... + an) / n)

""")



import time

totalstart = time.time()

def start(): start.start_time = time.time()

def end(): 
    end_time = time.time()
    return print("\nSuccess | Time elapsed:", round((end_time - start.start_time), 2), "seconds\n\n")



print("\n >>> Settings\n")

start()

random = 0

named_labels = [
"V00" ,
"V01" ,
"V02" ,
"V03" ,
"V04" ,
"V05" ,
"V06" ,
"V07" ,
"V08" ,
"V09" ,
"V10" ,
"V11" ,
"V12" ,
"V13" ,
"V14" ,
"V15" ,
"V16" ,
"V17" ,
"V18" ,
"V19" ,
"V20" ,
"V21" ,
"V22" ,
"V23" ,
"V24" ,
"V25" ,
"V26" ,
"V27" ,
"V28" ,
"V29" ,
"V30" ,
"V31" ,
"V32" ,
"V33" ,
"V34" ,
]

models = [
[ 1 , "LogisticRegression" ],
[ 2 ,     "NaiveBayes"     ],
[ 3 ,        "KNN"         ],
[ 4 ,    "DecisionTree"    ],
[ 5 ,        "SVM"         ],
[ 6 ,    "RandomForest"    ],
[ 7 ,      "XGBoost"       ],
[ 8 ,   "NeuralNetworks"   ],
]

threshold_high = 0.97
threshold_low  = 0.75

end()



print("\n >>> Importing tools\n")

start()

import os
import pickle
import operator
import numpy                 as np
import pandas                as pd
import random                as rd
import matplotlib.pyplot     as plt
from deap                    import base, creator, algorithms, tools
from scipy                   import stats
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from xgboost                 import XGBClassifier
from keras.models            import Sequential
from keras.layers            import Dense, Dropout
from keras.utils             import np_utils

end()



# Creating Results Directory

print("\n >>> Creating Results Directory")

start()

try: os.mkdir("results")
except OSError: pass

end()



# Importing Datasets

print("\n >>> Importing Datasets")

start()

dataset = pd.read_csv("frau.csv", sep = ";")
n_input = dataset.shape[1] - 1
labels  = ["v"+str(x) for x in range(n_input)]
x       = dataset[(labels)]
y       = dataset.M

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = random)

new_dataset = pd.read_csv("avaliar.csv", sep = ";")
new_x       = new_dataset[(labels)]

end()



### Variables Importances

print("\n >>> Variables Importances\n")

start()

forest = ExtraTreesClassifier(
	n_estimators             = 1000   ,
	criterion                ="gini"  ,
	max_depth                = None   ,
	min_samples_split        = 2      ,
	min_samples_leaf         = 1      ,
	min_weight_fraction_leaf = 0.0    ,
	max_features             = "auto" ,
	max_leaf_nodes           = None   ,
	min_impurity_decrease    = 0.0    ,
	min_impurity_split       = None   ,
	bootstrap                = False  ,
	oob_score                = False  ,
	n_jobs                   = None   ,
	random_state             = random ,
	verbose                  = 0      ,
	warm_start               = False  ,
	class_weight             = None   ,
	)

forest.fit(x_train, y_train)
importances = forest.feature_importances_

named_labels = named_labels[:n_input]

dic = dict(zip(named_labels, importances.round(4)))
sort_values = sorted(dic.items(), key = operator.itemgetter(1), reverse = False)
sorted_importances = pd.DataFrame(sort_values)

print(pd.DataFrame(sorted_importances.values, columns=["Variable", "Importance"]))



### Exporting Importances

lista = []

index = 0
for i in named_labels:
	lista.append(str(round(importances[index]*100,2)) + "% | " + str(i))
	index += 1

file = open('results\\importances.csv', 'w')

file.write('Importance|Variable\n')

index = 0
while index < len(named_labels):
	file.write(str(lista[index]) + '\n')
	index += 1

file.close()

print("File <importances.csv> saved.")

end()



# Genetic Algorithm Optimization

print("\n >>> Genetic Algorithm Optimization")

start()

dataset = pd.read_csv("frau.csv", sep = ";")

creator.create("FitnessMax", base.Fitness, weights = (1.0, ))
creator.create("Individual", list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.choice, a = [0, 1], p = [0.1, 0.9])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n = n_input)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def avaliacao(individual):
    
    labels = []
    counter = 0

    for x in individual:
        
        if x == 1: labels.append("v" + str(counter))
        else: pass
        counter += 1
    
    x = dataset[(labels)]
    y = dataset.M
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = random)

    MODEL = LogisticRegression(
        penalty           = "l2"    ,
        dual              = False   ,
        tol               = 0.0001  ,
        C                 = 1.0     ,
        fit_intercept     = True    ,
        intercept_scaling = 1       ,
        class_weight      = None    ,
        random_state      = random  ,
        solver            = "lbfgs" ,
        max_iter          = 100     ,
        multi_class       = "warn"  ,
        verbose           = 0       ,
        warm_start        = False   ,
        n_jobs            = None    ,
        l1_ratio          = None    ,
        )
        
    try:
        MODEL.fit(x_train, y_train)
        pred_y_test = MODEL.predict(x_test)
        accuracy = accuracy_score(y_test, pred_y_test)
        return [accuracy]
    
    except: return [0.0]

populacao = toolbox.population(n = 5) # Testar n = 1000
probabilidade_crossover = 0.05
probabilidade_mutacao = 0.2 # Testar 0.1
numero_geracoes = 5 # Testar 10

toolbox.register("evaluate", avaliacao)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb = probabilidade_crossover)
toolbox.register("select", tools.selRoulette)

estatisticas = tools.Statistics(key = lambda individuo: individuo.fitness.values)
estatisticas.register("min", np.min)
estatisticas.register("max", np.max)

hof = tools.HallOfFame(5)

print("\nOtimizando características do modelo com base em regressão logística...\n")

populacao, info = algorithms.eaSimple(population = populacao,
                                      toolbox    = toolbox,
                                      cxpb       = probabilidade_crossover,
                                      mutpb      = probabilidade_mutacao,
                                      ngen       = numero_geracoes,
                                      stats      = estatisticas,
                                      halloffame = hof,
                                      verbose    = True,
                                      )

print("\nOtimização realizada com sucesso!")

print("\nSequência com melhor assertividade:\n")

melhores = [int(i) for i in hof[0]]
print(melhores)

dic = dict(zip(labels, named_labels))

labels  = []
out     = []
counter = 0

for x in melhores:

    if x == 1: labels.append("v" + str(counter))
    else: out.append("v" + str(counter)) 
    counter += 1

print("\n >>> Variáveis utilizadas na solução otimizada\n")
for i in labels: print(i, dic[i])

print("\n >>> Variáveis que talvez estejam atrapalhando o modelo\n")
for i in out: print(i, dic[i])

def estudo(self):
    
    random = 0
    labels = []
    out = []
    counter = 0

    for x in self:
        
        if x == 1: labels.append("v" + str(counter))
        else: out.append(named_labels[counter]) 
        counter += 1
    
    x = dataset[(labels)]
    y = dataset.M
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = random)

    MODEL = LogisticRegression(
        penalty           = "l2"    ,
        dual              = False   ,
        tol               = 0.0001  ,
        C                 = 1.0     ,
        fit_intercept     = True    ,
        intercept_scaling = 1       ,
        class_weight      = None    ,
        random_state      = random  ,
        solver            = "lbfgs" ,
        max_iter          = 100     ,
        multi_class       = "warn"  ,
        verbose           = 0       ,
        warm_start        = False   ,
        n_jobs            = None    ,
        l1_ratio          = None    ,
        )
        
    MODEL.fit(x_train, y_train)
    pred_y_test = MODEL.predict(x_test)
    accuracy = accuracy_score(y_test, pred_y_test)
    
    return print("\nAssertividade do modelo de regressão logística otimizado:", round(accuracy * 100,4), "%")

estudo(melhores)

print("\nObs.: Esta seleção de variáveis ainda não foi colocada em produção para os próximos modelos.")

end()



# Models parameters

column_names     = []
accuracies       = []
scores           = []
cross_validation = pd.DataFrame() 

for i, j in models: 

    n_model         = i
    model_name      = j

    model           = "MODEL"           + "_" + str(n_model)
    pred_y_test     = "pred_y_test"     + "_" + str(n_model)
    accuracy        = "accuracy"        + "_" + str(n_model)
    pred_new_y      = "pred_new_y"      + "_" + str(n_model)
    new_probability = "new_probability" + "_" + str(n_model)
    score           = "score"           + "_" + str(n_model)

    column_name     = model + "_" + model_name

    print(">>> Running", column_name)

    start()

    if j == "XGBoost":
        
        x_train = x_train.values
        y_train = y_train.values
        x_test  =  x_test.values
        y_test  =  y_test.values
        new_x   =   new_x.values
    
    elif j == "NeuralNetworks":
        
        y_train_dummy = np_utils.to_categorical(y_train)
        y_test_dummy  = np_utils.to_categorical(y_test)    
    
    try: vars()[model] = pickle.load(open(model + ".sav", "rb"))

    except:

        if j == "LogisticRegression":
        
            vars()[model] = LogisticRegression(
                penalty           = "l2"    ,
                dual              = False   ,
                tol               = 0.0001  ,
                C                 = 1.0     ,
                fit_intercept     = True    ,
                intercept_scaling = 1       ,
                class_weight      = None    ,
                random_state      = random  ,
                solver            = "lbfgs" ,
                max_iter          = 100     ,
                multi_class       = "warn"  ,
                verbose           = 0       ,
                warm_start        = False   ,
                n_jobs            = None    ,
                l1_ratio          = None    ,
                )

        elif j == "NaiveBayes":
            
            vars()[model] = GaussianNB(
                priors        = None ,
                var_smoothing = 0.65 ,
                )

        elif j == "KNN":
        
            vars()[model] = KNeighborsClassifier(
                n_neighbors   = 5           ,
                weights       = "uniform"   ,
                algorithm     = "auto"      ,
                leaf_size     = 30          ,
                p             = 2           ,
                metric        = "minkowski" ,
                metric_params = None        ,
                n_jobs        = None        ,
                )

        elif j == "DecisionTree":            
            
            vars()[model] = DecisionTreeClassifier(
                criterion                = "gini" ,
                splitter                 = "best" ,
                max_depth                = None   ,
                min_samples_split        = 2      ,
                min_samples_leaf         = 1      ,
                min_weight_fraction_leaf = 0.0    ,
                max_features             = None   ,
                random_state             = random ,
                max_leaf_nodes           = None   ,
                min_impurity_decrease    = 0.0    ,
                min_impurity_split       = None   ,
                class_weight             = None   ,
                presort                  = False  ,
                )

        elif j == "SVM":
    
            vars()[model] = SVC(
                C                       = 1.0    ,
                kernel                  = "rbf"  ,
                degree                  = 3      ,
                gamma                   = "auto" ,
                coef0                   = 0.0    ,
                shrinking               = True   ,
                probability             = True   ,
                tol                     = 0.001  ,
                cache_size              = 200    ,
                class_weight            = None   ,
                verbose                 = False  ,
                max_iter                = -1     ,
                decision_function_shape = "ovr"  ,
                random_state            = random ,
                )

        elif j == "RandomForest":

            vars()[model] = RandomForestClassifier(
                n_estimators             = 1000   ,
                criterion                = "gini" ,
                max_depth                = None   ,
                min_samples_split        = 2      ,
                min_samples_leaf         = 1      ,
                min_weight_fraction_leaf = 0.0    ,
                max_features             = "auto" ,
                max_leaf_nodes           = None   ,
                min_impurity_decrease    = 0.0    ,
                min_impurity_split       = None   ,
                bootstrap                = True   ,
                oob_score                = False  ,
                n_jobs                   = None   ,
                random_state             = random ,
                verbose                  = 0      ,
                warm_start               = False  ,
                class_weight             = None   ,
                )

        elif j == "XGBoost":
         
            vars()[model] = XGBClassifier(
                base_score        = 0.5               ,
                booster           = "gbtree"          ,
                colsample_bylevel = 1                 ,
                colsample_bynode  = 1                 ,
                colsample_bytree  = 0.8               ,
                gamma             = 0                 ,
                learning_rate     = 0.2               ,
                max_delta_step    = 0                 ,         
                max_depth         = 5                 ,
                min_child_weight  = 1                 ,
                missing           = None              ,
                n_estimators      = 1000              ,
                n_jobs            = 1                 ,
                nthread           = None              ,
                objective         = "binary:logistic" ,
                random_state      = random            ,
                reg_alpha         = 0                 ,
                reg_lambda        = 1                 ,
                scale_pos_weight  = 1                 ,
                seed              = None              ,
                silent            = None              ,
                subsample         = 0.8               ,
                verbosity         = 1                 ,
                )

        elif j == "NeuralNetworks":
            
            vars()[model] = Sequential(
                layers = None,
                name   = None,
                )

            vars()[model].add(   Dense( units = 157 , activation = "relu", input_dim = n_input ))
            vars()[model].add( Dropout( rate  = 0.2                                            ))
            vars()[model].add(   Dense( units = 100 , activation = "relu"                      ))
            vars()[model].add(   Dense( units = 80  , activation = "relu"                      ))
            vars()[model].add(   Dense( units = 60  , activation = "relu"                      ))
            vars()[model].add(   Dense( units = 40  , activation = "relu"                      ))
            vars()[model].add(   Dense( units = 20  , activation = "relu"                      ))
            vars()[model].add(   Dense( units = 2   , activation = "softmax"                   ))

            vars()[model].compile(
                optimizer          = "adam"                     ,
                loss               = "categorical_crossentropy" ,
                metrics            = ["accuracy"]               ,
                loss_weights       = None                       ,
                sample_weight_mode = None                       ,
                weighted_metrics   = None                       ,
                target_tensors     = None                       ,
                )
    
    # Training (Fitting)
    
    if j != "NeuralNetworks" : vars()[model].fit(x_train, y_train)

    if j == "NeuralNetworks":
    
        vars()[model].fit(
            x_train                                      ,
            y_train_dummy                                ,
            batch_size          = None                   ,
            epochs              = 12                     ,
            verbose             = 0                      ,
            callbacks           = None                   ,
            validation_split    = 0.                     ,
            validation_data     = (x_test, y_test_dummy) ,
            shuffle             = True                   ,
            class_weight        = None                   ,
            sample_weight       = None                   ,
            initial_epoch       = 0                      ,               
            steps_per_epoch     = None                   ,
            validation_steps    = None                   ,
            validation_freq     = 1                      ,
            max_queue_size      = 10                     ,
            workers             = 1                      ,
            use_multiprocessing = False                  ,
            )
    
    pickle.dump(vars()[model], open(model + ".sav", "wb"))
    
    
    # Testing (Cross Validation)
        
    vars()[pred_y_test] = vars()[model].predict(x_test)
    
    cross_validation[column_name] = vars()[model].predict_proba(x_test)[:,1].round(4)
    
    if j != "NeuralNetworks": 
        
        vars()[accuracy] = accuracy_score(y_test, vars()[pred_y_test])
    
    if j == "NeuralNetworks": 
        
        pred_y_test_un   = [np.argmax(i) for i in vars()[pred_y_test]]
        vars()[accuracy] = accuracy_score(y_test, pred_y_test_un)
    
    print("\nAccuracy:", (vars()[accuracy] * 100).round(4), "%")
    
    
    # Predicting (Production)

    vars()[pred_new_y]      = vars()[model].predict(new_x)
    vars()[new_probability] = vars()[model].predict_proba(new_x).round(4)
    vars()[score]           = pd.DataFrame(vars()[new_probability][:,1], columns = [column_name])

    
    # Answers list creation
    
    column_names.append(column_name)
    accuracies.append(vars()[accuracy])
    scores.append(vars()[score])

    end()



# Ensemble Weights

print("\n >>> Ensemble Weights")
weights = pd.DataFrame(accuracies, columns = ["Weight"]).apply(lambda x: x.round(4))
weights.index = column_names 
print(weights)



# Maximum Average Score

max_avg_score = sum(accuracies) / len(models)

print("\n >>> Maximum Possible Average Score:", max_avg_score.round(4) ,"\n")



# Ensemble

print("\n >>> Results")

results = pd.DataFrame(new_dataset["t"]).join(scores)

def ensemble(self):
    
    self["MODEL_0_EnsembleLearning"] = (((
        sum([self[column_names[i]] * accuracies[i] for i in range(len(models))]) 
        / len(models)) 
        / max_avg_score).round(4))

ensemble(results)

results = results.sort_values(by="MODEL_0_EnsembleLearning", ascending=False)

results["STATUS_ML"] = results["MODEL_0_EnsembleLearning"].apply(lambda x: 
    
    "Automático (>= " + str(threshold_high) + ")" 
    if x >= threshold_high

    else "Mesa (>= " + str(threshold_low) + " e < " + str(threshold_high) + ")" 
    if x >= threshold_low 

    else "Não Fraude (< "+str(threshold_low)+")" 
    
    )

print(pd.DataFrame(results["STATUS_ML"].str.ljust(width = 24, fillchar = ' ').value_counts()))



# Exporting Results

print("\n >>> Exporting Results")

start()

results.to_csv(r"results.csv", index = False, sep = "|")
print("\nresults.csv exported!")

end()



# Cross validation

print("\n >>> Cross Validation")

print("""
> Assertividade (Accuracy)
Taxa de acertos para o que foi classificado como fraude e também para o que foi classificado como não fraude.

> Precisão (Precision, Positive Predictive Value (PPV))
Taxa de acertos para o que foi classificado como fraude.

> Sensibilidade (Sensitivity, True Positive Rate (TPR), Recall)
Taxa do volume de fraudes detectado. 

> Pontuação F1 (F1-Score)
Média harmônica entre Precisão e Sensibilidade.
""")

start()

ensemble(cross_validation)

cross_validation["Y_TEST"] = y_test

for j in [0.5, threshold_low, threshold_high]:

    print("\n\n >>> Corte = " + str(j) + "\n")
    
    cross_validation_binary = pd.DataFrame(cross_validation["Y_TEST"])
    
    fraud_vision = []
    
    for i in list(cross_validation.columns[:-1]):    
        
        cross_validation_binary[i] = cross_validation[i].apply(lambda x: 1 if x >= j else 0)
        
        accuracy    = str((accuracy_score  (cross_validation_binary["Y_TEST"], cross_validation_binary[i]) * 100).round(2)) + "%"
        
        precision   = str((precision_score (cross_validation_binary["Y_TEST"], cross_validation_binary[i]) * 100).round(2)) + "%"
        
        sensitivity = str((recall_score    (cross_validation_binary["Y_TEST"], cross_validation_binary[i]) * 100).round(2)) + "%"
        
        f1          = str((f1_score        (cross_validation_binary["Y_TEST"], cross_validation_binary[i]) * 100).round(2)) + "%"
               
        fraud_vision.append([i[6:], 
                             accuracy, 
                             precision, 
                             sensitivity, 
                             f1,
                            ])
      
    print(pd.DataFrame(fraud_vision, 
                       columns = [
                           "Modelo", 
                           "Accuracy", 
                           "Precision", 
                           "Sensitivity",
                           "F1-Score",
                       ]).set_index("Modelo"), "\n")
    
end()



# ROC Curve & AUC Score

print("\n >>> ROC Curve & AUC Score")

start()

plt.rcParams['figure.figsize'] = 15, 9

cross_validation_binary = cross_validation

areas = []

for i in list(cross_validation.columns[:-1]):
    
    fpr, tpr, thresholds = roc_curve(cross_validation_binary["Y_TEST"], cross_validation_binary[i])
    
    auc = str((roc_auc_score(cross_validation_binary["Y_TEST"], cross_validation_binary[i]) * 100).round(2)) + "%"
    areas.append("AUC Score = " + auc)
    
    plt.plot(fpr, tpr)
    
fpr        = fpr        .round(2)
tpr        = tpr        .round(2)
thresholds = thresholds .round(4)

cutoffs = pd.DataFrame(zip(fpr, tpr, thresholds), columns = ["FPR","TPR","Corte"]).drop(0)

cutoff_default = cutoffs[cutoffs.Corte.round(3) == 0.5            ].head(1)
cutoff_low     = cutoffs[cutoffs.Corte.round(3) == threshold_low  ].head(1)
cutoff_high    = cutoffs[cutoffs.Corte.round(3) == threshold_high ].head(1)

annotations = cutoff_default.append(cutoff_low).append(cutoff_high)
    
for x, y, cutoff in list(annotations.values):
    plt.annotate(cutoff.round(2), 
                 (x, y), 
                 fontsize=10, 
                 weight='bold',
                 xytext= (x-0.05, y+0.02),
                 arrowprops=dict(arrowstyle="simple",fc="0.6", ec="none", connectionstyle="arc3,rad=-0.2"),
                )
    
plt.legend(list(zip(areas, list(cross_validation.columns[:-1]))),loc='lower center', fontsize=15)
plt.title('\nROC Curve & AUC Score\n', fontsize=20, weight='bold')
plt.xlabel('\nFPR\n', fontsize=17, weight='bold')
plt.ylabel('\nTPR\n', fontsize=17, weight='bold')
plt.grid(alpha=0.5)    
plt.tight_layout()
plt.show()

end()



# Time

totalend = time.time()

print("Total time elapsed:", round((totalend - totalstart),2), "seconds\n\n")



# End

input("\nPress ENTER to exit")