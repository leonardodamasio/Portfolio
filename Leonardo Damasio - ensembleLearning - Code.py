print("""\

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

""")



print("\n*** Importing tools ***\n")

import time

totalstart = time.time()

start = time.time()

import pandas                as pd
import numpy                 as np
import matplotlib.pyplot     as plt
import operator
import pickle
import os
from scipy                   import stats
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, confusion_matrix
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

end = time.time()

print("\nSuccess\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# Creating Results Directory

print("\n*** Creating Results Directory ***")

start = time.time()

try: os.mkdir("results")
except OSError: pass

end = time.time()

print("\nSuccess\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# Importing Datasets

print("\n*** Importing Datasets ***")

start = time.time()

n_input = 26
labels = ["v"+str(x) for x in range(n_input)]
random = 0

dataset = pd.read_csv("frau.csv", sep = ";")
x = dataset[(labels)]
y = dataset.M
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random)

new_dataset = pd.read_csv("avaliar.csv", sep = ";")
new_x = new_dataset[(labels)]

end = time.time()

print("\nSuccess\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



### Variables Importances

print("\n*** Variables Importances ***")

start = time.time()

forest = ExtraTreesClassifier(
	n_estimators=1000,
	criterion="gini",
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0,
	max_features="auto",
	max_leaf_nodes=None,
	min_impurity_decrease=0.0,
	min_impurity_split=None,
	bootstrap=False,
	oob_score=False,
	n_jobs=None,
	random_state=random,
	verbose=0,
	warm_start=False,
	class_weight=None
	)

forest.fit(x_train, y_train)
importances = forest.feature_importances_

named_labels = [
"VAR_00",
"VAR_01",
"VAR_02",
"VAR_03",
"VAR_04",
"VAR_05",
"VAR_06",
"VAR_07",
"VAR_08",
"VAR_09",
"VAR_10",
"VAR_11",
"VAR_12",
"VAR_13",
"VAR_14",
"VAR_15",
"VAR_16",
"VAR_17",
"VAR_18",
"VAR_19",
"VAR_20",
"VAR_21",
"VAR_22",
"VAR_23",
"VAR_24",
"VAR_25"
]

dic = dict(zip(named_labels, importances.round(4)))
sort_values = sorted(dic.items(), key=operator.itemgetter(1), reverse=False)
sorted_importances = pd.DataFrame(sort_values)

print("\nVariables Importances\n")
print(pd.DataFrame(sorted_importances.values, columns=["Variable", "Importance"]))



### Variables Importances Plot

plt.rcParams['figure.figsize'] = 12, 10
plt.scatter(sorted_importances[1], sorted_importances[0])
plt.title('\nImportances\n', fontsize=20)
plt.xlabel('\nImportance (0~1)\n', fontsize=15)
plt.ylabel('\nVariable\n', fontsize=15)
plt.grid(alpha=0.5)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('results\\importances.png', format='png', dpi = 300, bbox_inches='tight')
# plt.show()

print("\nImage <importances.png> saved.\n")



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
	file.write(str(lista[index])+'\n')
	index += 1

file.close()

print("File <importances.csv> saved.\n")

end = time.time()

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# Training, Testing and Predicting


# MODEL_1 Logistic Regression

print("\n*** Running MODEL_1 Logistic Regression ***")

start = time.time()

try: MODEL_1 = pickle.load(open("MODEL_1.sav", "rb"))

except:

	MODEL_1 = LogisticRegression(
		penalty="l2",
		dual=False,
		tol=0.0001,
		C=1.0,
		fit_intercept=True,
		intercept_scaling=1,
		class_weight=None,
		random_state=random,
		solver="lbfgs",
		max_iter=100,
		multi_class="warn",
		verbose=0,
		warm_start=False,
		n_jobs=None,
		l1_ratio=None
		)

	MODEL_1.fit(x_train, y_train)

	pickle.dump(MODEL_1, open("MODEL_1.sav", "wb"))

pred_y_test_1 = MODEL_1.predict(x_test)
accuracy_1 = accuracy_score(y_test, pred_y_test_1)

pred_new_y = MODEL_1.predict(new_x)
new_probability = MODEL_1.predict_proba(new_x).round(4)
scores_1 = pd.DataFrame(new_probability[:,1], columns=["MODEL_1_LogisticRegression"])

end = time.time()

print("\nMODEL_1/8: Logistic Regression | Accuracy:", (accuracy_1*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")


# MODEL_2 Naive Bayes Classifier

print("\n*** Running MODEL_2 Naive Bayes Classifier ***")

start = time.time()

try: MODEL_2 = pickle.load(open("MODEL_2.sav", "rb"))

except:

	MODEL_2 = GaussianNB(
		priors=None,
		var_smoothing=0.65
		)

	MODEL_2.fit(x_train, y_train)

	pickle.dump(MODEL_2, open("MODEL_2.sav", "wb"))

pred_y_test_2 = MODEL_2.predict(x_test)
accuracy_2 = accuracy_score(y_test, pred_y_test_2)

pred_new_y = MODEL_2.predict(new_x)
new_probability = MODEL_2.predict_proba(new_x).round(4)
scores_2 = pd.DataFrame(new_probability[:,1], columns=["MODEL_2_NaiveBayes"])

end = time.time()

print("\nMODEL_2/8: Naive Bayes Classifier | Accuracy:", (accuracy_2*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# MODEL_3 K-Nearest Neighbors Classifier

print("\n*** Running MODEL_3 K-Nearest Neighbors Classifier ***")

start = time.time()

try: MODEL_3 = pickle.load(open("MODEL_3.sav", "rb"))

except:

	MODEL_3 = KNeighborsClassifier(
		n_neighbors=5,
		weights="uniform",
		algorithm="auto",
		leaf_size=30,
		p=2,
		metric="minkowski",
		metric_params=None,
		n_jobs=None
		)

	MODEL_3.fit(x_train, y_train)

	pickle.dump(MODEL_3, open("MODEL_3.sav", "wb"))

pred_y_test_3 = MODEL_3.predict(x_test)
accuracy_3 = accuracy_score(y_test, pred_y_test_3)

pred_new_y = MODEL_3.predict(new_x)
new_probability = MODEL_3.predict_proba(new_x).round(4)
scores_3 = pd.DataFrame(new_probability[:,1], columns=["MODEL_3_KNN"])

end = time.time()

print("\nMODEL_3/8: K-Nearest Neighbors Classifier | Accuracy:", (accuracy_3*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# MODEL_4 Decision Tree Classifier

print("\n*** Running MODEL_4 Decision Tree Classifier ***")

start = time.time()

try: MODEL_4 = pickle.load(open("MODEL_4.sav", "rb"))

except:

	MODEL_4 = DecisionTreeClassifier(
		criterion="gini",
		splitter="best",
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		min_weight_fraction_leaf=0.0,
		max_features=None,
		random_state=random,
		max_leaf_nodes=None,
		min_impurity_decrease=0.0,
		min_impurity_split=None,
		class_weight=None,
		presort=False
		)

	MODEL_4.fit(x_train, y_train)

	pickle.dump(MODEL_4, open("MODEL_4.sav", "wb"))

pred_y_test_4 = MODEL_4.predict(x_test)
accuracy_4 = accuracy_score(y_test, pred_y_test_4)

pred_new_y = MODEL_4.predict(new_x)
new_probability = MODEL_4.predict_proba(new_x).round(4)
scores_4 = pd.DataFrame(new_probability[:,1], columns=["MODEL_4_DecisionTree"])

end = time.time()

print("\nMODEL_4/8: Decision Tree Classifier | Accuracy:", (accuracy_4*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# MODEL_5 Support Vector Machine Classifier

print("\n*** Running MODEL_5 Support Vector Machine Classifier ***")

start = time.time()

try: MODEL_5 = pickle.load(open("MODEL_5.sav", "rb"))

except:

	MODEL_5 = SVC(
		C=1.0,
		kernel="rbf",
		degree=3,
		gamma="auto",
		coef0=0.0,
		shrinking=True,
		probability=True,
		tol=0.001,
		cache_size=200,
		class_weight=None,
		verbose=False,
		max_iter=-1,
		decision_function_shape="ovr",
		random_state=random
		)

	MODEL_5.fit(x_train, y_train)

	pickle.dump(MODEL_5, open("MODEL_5.sav", "wb"))

pred_y_test_5 = MODEL_5.predict(x_test)
accuracy_5 = accuracy_score(y_test, pred_y_test_5)

pred_new_y = MODEL_5.predict(new_x)
new_probability = MODEL_5.predict_proba(new_x).round(4)
scores_5 = pd.DataFrame(new_probability[:,1], columns=["MODEL_5_SVM"])

end = time.time()

print("\nMODEL_5/8: Support Vector Machine Classifier | Accuracy:", (accuracy_5*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# MODEL_6 Random Forest Classifier

print("\n*** Running MODEL_6 Random Forest Classifier ***")

start = time.time()

try: MODEL_6 = pickle.load(open("MODEL_6.sav", "rb"))

except:

	MODEL_6 = RandomForestClassifier(
		n_estimators=1000,
		criterion="gini",
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		min_weight_fraction_leaf=0.0,
		max_features="auto",
		max_leaf_nodes=None,
		min_impurity_decrease=0.0,
		min_impurity_split=None,
		bootstrap=True,
		oob_score=False,
		n_jobs=None,
		random_state=random,
		verbose=0,
		warm_start=False,
		class_weight=None
		)

	MODEL_6.fit(x_train, y_train)

	pickle.dump(MODEL_6, open("MODEL_6.sav", "wb"))

pred_y_test_6 = MODEL_6.predict(x_test)
accuracy_6 = accuracy_score(y_test, pred_y_test_6)

pred_new_y = MODEL_6.predict(new_x)
new_probability = MODEL_6.predict_proba(new_x).round(4)
scores_6 = pd.DataFrame(new_probability[:,1], columns=["MODEL_6_RandomForest"])

end = time.time()

print("\nMODEL_6/8: Random Forest Classifier | Accuracy:", (accuracy_6*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# MODEL_7 Extreme Gradient Boosted Trees Classifier

print("\n*** Running MODEL_7 Extreme Gradient Boosted Trees Classifier ***")

start = time.time()

x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values
new_x = new_x.values

try: MODEL_7 = pickle.load(open("MODEL_7.sav", "rb"))

except:

	MODEL_7 = XGBClassifier(
		base_score=0.5,
		booster="gbtree",
		colsample_bylevel=1,
		colsample_bynode=1,
		colsample_bytree=0.8,
		gamma=0,
		learning_rate=0.2,
		max_delta_step=0,
		max_depth=5,
		min_child_weight=1,
		missing=None,
		n_estimators=1000,
		n_jobs=1,
		nthread=None,
		objective="binary:logistic",
		random_state=random,
		reg_alpha=0,
		reg_lambda=1,
		scale_pos_weight=1,
		seed=None,
		silent=None,
		subsample=0.8,
		verbosity=1
		)

	MODEL_7.fit(x_train, y_train)

	pickle.dump(MODEL_7, open("MODEL_7.sav", "wb"))

pred_y_test_7 = MODEL_7.predict(x_test)
accuracy_7 = accuracy_score(y_test, pred_y_test_7)

pred_new_y = MODEL_7.predict(new_x)
new_probability = MODEL_7.predict_proba(new_x).round(4)
scores_7 = pd.DataFrame(new_probability[:,1], columns=["MODEL_7_XGBoost"])

end = time.time()

print("\nMODEL_7/8: Extreme Gradient Boosted Trees Classifier | Accuracy:", (accuracy_7*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# MODEL_8 Deep Learning Multilayer Perceptron Neural Networks

print("\n*** Running MODEL_8 Deep Learning Multilayer Perceptron Neural Networks ***")

start = time.time()

y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)

try: MODEL_8 = pickle.load(open("MODEL_8.sav", "rb"))

except:

	MODEL_8 = Sequential(
		layers=None,
		name=None
		)

	MODEL_8.add(Dense(units=157, activation="relu", input_dim=n_input))
	MODEL_8.add(Dropout(rate=0.2))
	MODEL_8.add(Dense(units=100, activation="relu"))
	MODEL_8.add(Dense(units=80, activation="relu"))
	MODEL_8.add(Dense(units=60, activation="relu"))
	MODEL_8.add(Dense(units=40, activation="relu"))
	MODEL_8.add(Dense(units=20, activation="relu"))
	MODEL_8.add(Dense(units=2, activation="softmax"))

	MODEL_8.compile(
		optimizer="adam",
		loss="categorical_crossentropy",
		metrics=["accuracy"],
		loss_weights=None,
		sample_weight_mode=None,
		weighted_metrics=None,
		target_tensors=None
		)

	MODEL_8.fit(
		x_train,
		y_train,
		batch_size=None,
		epochs=12,
		verbose=0,
		callbacks=None,
		validation_split=0.,
		validation_data=(x_test, y_test),
		shuffle=True,
		class_weight=None,
		sample_weight=None,
		initial_epoch=0,
		steps_per_epoch=None,
		validation_steps=None,
		validation_freq=1,
		max_queue_size=10,
		workers=1,
		use_multiprocessing=False
		)

	pickle.dump(MODEL_8, open("MODEL_8.sav", "wb"))

pred_y_test_8 = MODEL_8.predict(x_test)
y_test_un = [np.argmax(i) for i in y_test]
pred_y_test_8_un = [np.argmax(i) for i in pred_y_test_8]
accuracy_8 = accuracy_score(y_test_un, pred_y_test_8_un)

pred_new_y = MODEL_8.predict(new_x)
new_probability = MODEL_8.predict_proba(new_x).round(4)
scores_8 = pd.DataFrame(new_probability[:,1], columns=["MODEL_8_NeuralNetworks"])

end = time.time()

print("\nMODEL_8/8: Deep Learning Multilayer Perceptron Neural Networks | Accuracy:", (accuracy_8*100).round(4), "%\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# Ensemble Weights

print("\n*** Ensemble Weights ***")

print("\nMODEL_1/8: Logistic Regression                                 | Accuracy:", (accuracy_1*100).round(4), "%")
print("\nMODEL_2/8: Naive Bayes Classifier                              | Accuracy:", (accuracy_2*100).round(4), "%")
print("\nMODEL_3/8: K-Nearest Neighbors Classifier                      | Accuracy:", (accuracy_3*100).round(4), "%")
print("\nMODEL_4/8: Decision Tree Classifier                            | Accuracy:", (accuracy_4*100).round(4), "%")
print("\nMODEL_5/8: Support Vector Machine Classifier                   | Accuracy:", (accuracy_5*100).round(4), "%")
print("\nMODEL_6/8: Random Forest Classifier                            | Accuracy:", (accuracy_6*100).round(4), "%")
print("\nMODEL_7/8: Extreme Gradient Boosted Trees Classifier           | Accuracy:", (accuracy_7*100).round(4), "%")
print("\nMODEL_8/8: Deep Learning Multilayer Perceptron Neural Networks | Accuracy:", (accuracy_8*100).round(4), "%")



# Maximum Average Score

max_avg_score = (accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4 + accuracy_5 + accuracy_6 + accuracy_7 + accuracy_8) / 8
print("\n\n\n*** Maximum Possible Average Score ***")
print("\n",max_avg_score,"\n")



# Final Score

resultados = pd.DataFrame(new_dataset["t"]).join([scores_1, scores_2, scores_3, scores_4, scores_5, scores_6, scores_7, scores_8])

resultados["MODEL_0_EnsembleLearning"] = (((
	resultados["MODEL_1_LogisticRegression"] * accuracy_1 + 
	resultados["MODEL_2_NaiveBayes"]         * accuracy_2 +
	resultados["MODEL_3_KNN"]                * accuracy_3 +
	resultados["MODEL_4_DecisionTree"]       * accuracy_4 +
	resultados["MODEL_5_SVM"]                * accuracy_5 +
	resultados["MODEL_6_RandomForest"]       * accuracy_6 +
	resultados["MODEL_7_XGBoost"]            * accuracy_7 +
	resultados["MODEL_8_NeuralNetworks"]     * accuracy_8) / 8 ) / max_avg_score).round(4)



# Exporting Results

print("\n\n*** Exporting Results ***")

start = time.time()

resultados = resultados.sort_values(by="MODEL_0_EnsembleLearning", ascending=False)
resultados.to_csv(r"results.csv", index=False, sep="|")
print("\nresults.csv exported!")

para_massivo = resultados[resultados["MODEL_0_EnsembleLearning"] >= 0.97].sort_values(by="MODEL_0_EnsembleLearning", ascending=False)
para_massivo.to_csv(r"massivo.csv", index=False, sep="|")
print("\nmassivo.csv exported!")

para_mesa = resultados[(resultados["MODEL_0_EnsembleLearning"] >= 0.75) & (resultados["MODEL_0_EnsembleLearning"] < 0.97)].sort_values(by="MODEL_0_EnsembleLearning", ascending=False)
para_mesa.to_csv(r"mesa.csv", index=False, sep="|")
print("\nmesa.csv exported!")

end = time.time()

print("\nSuccess\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# Backtest

print("\n*** Backtest ***")

start = time.time()

backtest = pd.DataFrame()

backtest["MODEL_1_LogisticRegression"] = MODEL_1.predict_proba(x_test)[:,1].round(4)
backtest["MODEL_2_NaiveBayes"]         = MODEL_2.predict_proba(x_test)[:,1].round(4)
backtest["MODEL_3_KNN"]                = MODEL_3.predict_proba(x_test)[:,1].round(4)
backtest["MODEL_4_DecisionTree"]       = MODEL_4.predict_proba(x_test)[:,1].round(4)
backtest["MODEL_5_SVM"]                = MODEL_5.predict_proba(x_test)[:,1].round(4)
backtest["MODEL_6_RandomForest"]       = MODEL_6.predict_proba(x_test)[:,1].round(4)
backtest["MODEL_7_XGBoost"]            = MODEL_7.predict_proba(x_test)[:,1].round(4)
backtest["MODEL_8_NeuralNetworks"]     = MODEL_8.predict_proba(x_test)[:,1].round(4)

backtest["MODEL_0_EnsembleLearning"] = (((
	backtest["MODEL_1_LogisticRegression"] * accuracy_1 + 
	backtest["MODEL_2_NaiveBayes"]         * accuracy_2 +
	backtest["MODEL_3_KNN"]                * accuracy_3 +
	backtest["MODEL_4_DecisionTree"]       * accuracy_4 +
	backtest["MODEL_5_SVM"]                * accuracy_5 +
	backtest["MODEL_6_RandomForest"]       * accuracy_6 +
	backtest["MODEL_7_XGBoost"]            * accuracy_7 +
	backtest["MODEL_8_NeuralNetworks"]     * accuracy_8) / 8 ) / max_avg_score).round(4)

backtest["Y_TEST"] = y_test_un

for i in list(backtest.columns)[:-1]:

    backtest_binario = pd.DataFrame()

    backtest_binario[i] = backtest[i].apply(

        lambda x: 1 if 

#         x >= 0.5

#         x >= 0.75
#         and 
#         x < 0.97

        x >= 0.97

        else 0
    )

    backtest_binario["Y_TEST"] = backtest["Y_TEST"]

    backtest_binario = backtest_binario[backtest_binario[i] == 1]

    print("\n***", i, "***\n\n", (accuracy_score(backtest_binario["Y_TEST"], backtest_binario[i])*100).round(2), "% de assertividade\n")

    print("", confusion_matrix(backtest_binario["Y_TEST"], backtest_binario[i])[1][1], "fraudes encontradas")
    print("", confusion_matrix(backtest_binario["Y_TEST"], backtest_binario[i])[0][1], "falsos positivos\n")

    print(backtest_binario.head(),"\n")

end = time.time()

print("\nSuccess\n")

print("Time elapsed:", round((end - start),2), "seconds\n\n")



# Time

totalend = time.time()

print("Total time elapsed:", round((totalend - totalstart),2), "seconds\n\n")



# End

input("\nPress ENTER to exit")