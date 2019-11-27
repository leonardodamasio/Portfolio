
Ensemble Learning
Objetivo: Detecção de fraude

Autor: Leonardo Damasio | Senior Data Scientist
Data: 27/11/2019

                                                                                                
FFFFFFF RRRRRR    AAA   UU   UU DDDDD      DDDDD   EEEEEEE TTTTTTT EEEEEEE  CCCCC  TTTTTTT IIIII  OOOOO  NN   NN 
FF      RR   RR  AAAAA  UU   UU DD  DD     DD  DD  EE        TTT   EE      CC    C   TTT    III  OO   OO NNN  NN 
FFFF    RRRRRR  AA   AA UU   UU DD   DD    DD   DD EEEEE     TTT   EEEEE   CC        TTT    III  OO   OO NN N NN 
FF      RR  RR  AAAAAAA UU   UU DD   DD    DD   DD EE        TTT   EE      CC    C   TTT    III  OO   OO NN  NNN 
FF      RR   RR AA   AA  UUUUU  DDDDDD     DDDDDD  EEEEEEE   TTT   EEEEEEE  CCCCC    TTT   IIIII  OOOO0  NN   NN 
                                                                                                                 
                                                                                             by Leonardo Damasio



*** Ensemble Learning ***

What is an ensemble method?

"Ensemble models in machine learning combine the decisions from multiple models to improve the overall performance." 

                                                                                    Reference: Towards Data Science



*** Models Used ***

Logistic Regression
Naive Bayes Classifier
K-Nearest Neighbors Classifier
Decision Tree Classifier
Support Vector Machine Classifier
Random Forest Classifier
Extreme Gradient Boosted Trees Classifier
Deep Learning Multilayer Perceptron Neural Networks



*** Ensemble Technique ***

Model Score = m
Accuracy = a
Final Score = Weighted Average = ((m1 * a1 + m2 * a2 + ... + mn * an) / n) / ((a1 + a2 + ... + an) / n)



*** Importing tools ***

Using TensorFlow backend.

Success

Time elapsed: 10.14 seconds



*** Creating Results Directory ***

Success

Time elapsed: 0.0 seconds



*** Importing Datasets ***

Success

Time elapsed: 0.47 seconds



*** Variables Importances ***

Variables Importances

   Variable Importance
0    VAR_00     0.0007
1    VAR_21     0.0012
2    VAR_19     0.0017
3    VAR_18     0.0021
4    VAR_20     0.0028
5    VAR_25     0.0031
6    VAR_22     0.0035
7    VAR_03     0.0038
8    VAR_02     0.0048
9    VAR_11     0.0061
10   VAR_12     0.0065
11   VAR_07     0.0081
12   VAR_16     0.0099
13   VAR_08     0.0106
14   VAR_04     0.0108
15   VAR_14     0.0112
16   VAR_01     0.0115
17   VAR_15     0.0127
18   VAR_13      0.013
19   VAR_17     0.0297
20   VAR_24     0.0367
21   VAR_23     0.0391
22   VAR_10     0.0436
23   VAR_09     0.0631
24   VAR_06     0.1642
25   VAR_05     0.4994

Image <importances.png> saved.

File <importances.csv> saved.

Time elapsed: 100.49 seconds



*** Running MODEL_1 Logistic Regression ***

MODEL_1/8: Logistic Regression | Accuracy: 92.3843 %

Time elapsed: 0.03 seconds



*** Running MODEL_2 Naive Bayes Classifier ***

MODEL_2/8: Naive Bayes Classifier | Accuracy: 91.2391 %

Time elapsed: 0.09 seconds



*** Running MODEL_3 K-Nearest Neighbors Classifier ***

MODEL_3/8: K-Nearest Neighbors Classifier | Accuracy: 92.8772 %

Time elapsed: 268.13 seconds



*** Running MODEL_4 Decision Tree Classifier ***

MODEL_4/8: Decision Tree Classifier | Accuracy: 93.1138 %

Time elapsed: 0.03 seconds



*** Running MODEL_5 Support Vector Machine Classifier ***

MODEL_5/8: Support Vector Machine Classifier | Accuracy: 93.0664 %

Time elapsed: 57.46 seconds



*** Running MODEL_6 Random Forest Classifier ***

MODEL_6/8: Random Forest Classifier | Accuracy: 93.4548 %

Time elapsed: 16.42 seconds



*** Running MODEL_7 Extreme Gradient Boosted Trees Classifier ***

MODEL_7/8: Extreme Gradient Boosted Trees Classifier | Accuracy: 93.5693 %

Time elapsed: 1.53 seconds



*** Running MODEL_8 Deep Learning Multilayer Perceptron Neural Networks ***

MODEL_8/8: Deep Learning Multilayer Perceptron Neural Networks | Accuracy: 93.4673 %

Time elapsed: 81.55 seconds



*** Ensemble Weights ***

MODEL_1/8: Logistic Regression                                 | Accuracy: 92.3843 %

MODEL_2/8: Naive Bayes Classifier                              | Accuracy: 91.2391 %

MODEL_3/8: K-Nearest Neighbors Classifier                      | Accuracy: 92.8772 %

MODEL_4/8: Decision Tree Classifier                            | Accuracy: 93.1138 %

MODEL_5/8: Support Vector Machine Classifier                   | Accuracy: 93.0664 %

MODEL_6/8: Random Forest Classifier                            | Accuracy: 93.4548 %

MODEL_7/8: Extreme Gradient Boosted Trees Classifier           | Accuracy: 93.5693 %

MODEL_8/8: Deep Learning Multilayer Perceptron Neural Networks | Accuracy: 93.4673 %



*** Maximum Possible Average Score ***

 0.9289653197898773 



*** Exporting Results ***

results.csv exported!

massivo.csv exported!

mesa.csv exported!

Success

Time elapsed: 0.41 seconds



*** Backtest ***

*** MODEL_1_LogisticRegression ***

 98.8 % de assertividade

 17733 fraudes encontradas
 215 falsos positivos

    MODEL_1_LogisticRegression  Y_TEST
0                            1       0
1                            1       1
5                            1       1
11                           1       1
14                           1       1 


*** MODEL_2_NaiveBayes ***

 99.51 % de assertividade

 14880 fraudes encontradas
 73 falsos positivos

    MODEL_2_NaiveBayes  Y_TEST
0                    1       0
1                    1       1
5                    1       1
15                   1       1
19                   1       1 


*** MODEL_3_KNN ***

 97.51 % de assertividade

 21267 fraudes encontradas
 544 falsos positivos

    MODEL_3_KNN  Y_TEST
0             1       0
1             1       1
5             1       1
8             1       1
10            1       0 


*** MODEL_4_DecisionTree ***

 97.98 % de assertividade

 20064 fraudes encontradas
 413 falsos positivos

   MODEL_4_DecisionTree  Y_TEST
0                     1       0
1                     1       1
3                     1       0
5                     1       1
8                     1       1 


*** MODEL_5_SVM ***

 99.27 % de assertividade

 12976 fraudes encontradas
 95 falsos positivos

    MODEL_5_SVM  Y_TEST
1             1       1
5             1       1
11            1       1
15            1       1
19            1       1 


*** MODEL_6_RandomForest ***

 98.95 % de assertividade

 19144 fraudes encontradas
 203 falsos positivos

   MODEL_6_RandomForest  Y_TEST
0                     1       0
1                     1       1
3                     1       0
5                     1       1
8                     1       1 


*** MODEL_7_XGBoost ***

 99.15 % de assertividade

 18663 fraudes encontradas
 160 falsos positivos

    MODEL_7_XGBoost  Y_TEST
0                 1       0
1                 1       1
5                 1       1
8                 1       1
11                1       1 


*** MODEL_8_NeuralNetworks ***

 99.18 % de assertividade

 18815 fraudes encontradas
 156 falsos positivos

    MODEL_8_NeuralNetworks  Y_TEST
0                        1       0
1                        1       1
5                        1       1
8                        1       1
11                       1       1 


*** MODEL_0_EnsembleLearning ***

 99.34 % de assertividade

 16809 fraudes encontradas
 111 falsos positivos

    MODEL_0_EnsembleLearning  Y_TEST
0                          1       0
1                          1       1
5                          1       1
11                         1       1
14                         1       1 


Success

Time elapsed: 147.82 seconds


Total time elapsed: 684.58 seconds



Press ENTER to exit