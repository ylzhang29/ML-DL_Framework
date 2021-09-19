
import os
import csv
import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, Lasso,ElasticNet,LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF,DotProduct
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV, PredefinedSplit,permutation_test_score, validation_curve
from sklearn.model_selection import permutation_test_score, PredefinedSplit

import sys

# file to save the search results
sys.stdout = open("/.../...Search_results.csv", "w")

# IMPORT DATA: TARGET CATEGORY MUST BE IN LAST COLUMN, FEATURE NAMES IN FIRST ROW
os.chdir("/.../.../xxx")
data = np.loadtxt('xxxx.csv', delimiter=",", skiprows=1)
# SELECT FEATURES AND TARGET FROM DATA
features = data[:, 1:-5]
print("feature shape", features.shape)
target = data[:, -2]

#define train and validation subsets
subset = data[:, -1] # training set is -1; validation set is 0
ps = PredefinedSplit(test_fold = subset)
ps.get_n_splits()
print(ps)
for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
# alternative cross-validation strategy e.g:
# cv = StratifiedKFold(n_splits=3, shuffle=True) # see blow

#DEFINE CLASSIFICATION ESTIMATORS
def choose_model(model = "mlp"):
    if model == "mlp" :
        parameters = {'hidden_layer_sizes': [(10), (10,10), (10,10,10), (10, 10, 10, 10, 10), (10, 10, 10, 10, 10, 10, 10, 10)], 'alpha': [.00001, 0.0001, 0.001],
                      'batch_size': ['auto'], 'learning_rate_init': [0.0001, 0.0005,.001, 0.01], }
        model_name = "MLP"
    elif model == "kn" :
        parameters = {'n_neighbors': [5,10,30,40,50,100,500], 'leaf_size': [5,10,15,20,30,50,100],
                      'p': [1, 2]}
        model_name = "k Nearest Neighbors"
    elif model == "rf":
        parameters = {'max_features': [.1,  .2, .3, .4, .5 ],
                      'n_estimators': [20, 50, 100, 500, 1000 ], 'max_depth' : [1,3,5,10, 15, 20]}
        model_name = "Random Forests"
    elif model == "sv":
        parameters = {'C': [.01,  1, 10, 100,  1000, 10000],
                      'gamma': [.01, .1, 1, 10, 100, 1000, 10000]}
        model_name = "Support Vector Machine"
    elif model == "gb":
        parameters = {'learning_rate': [0.001, .005, 0.01, 0.05, 0.1],
                      'n_estimators': [20,50, 100, 500, 1000, 5000, 10000],
                      'max_depth': [1, 2, 3, 5, 10]}
        model_name = "Gradient Boosting"
    elif model == "gp":
        parameters = {'kernel': [1.0 * RBF(length_scale=1.0), 1.0 * DotProduct(sigma_0=1.0)**2]}
        model_name = "Gaussian Process Classifier"
    elif model == "dt":
        parameters = {'criterion': ["gini", "entropy"], 'splitter': ["best", "random"],'max_depth': [1,3,5,10,15, 20, 30,50],
                      'max_features': [.1, .2, .3, .4,.5]}
        model_name = "Decision Tree Classifier"
    elif model == "gnb":
        parameters = {} #'var_smoothing': [0.000000001]
        model_name = "Gaussian Naive Bayes"
    elif model == "qda":
        parameters = {'reg_param': [.0, .00001, .0001], 'tol': [0.00001, 0.0001, 0.001]}
        model_name = "Quadratic Discriminant Analysis"
    elif model == "nsv":
        parameters = {'tol': [0.0001, 0.001, .01], 'kernel': ["linear","rbf"],# "sigmoid", "poly"], 
                      'gamma': [.0001, .001, .01, .1, 1, 10, 100, 500, 1000], 'nu': [ .1, .2, .5, .75]}
        model_name = "Nu SVC"
    elif model == "lsv":
        parameters = {'C': [.01,  1, 10, 100,  1000, 10000],'loss':['squared_hinge'], 'tol': [0.00001,0.0001, 0.001],
                      'penalty': ['l2']}
        model_name = "Linear SVC"
    elif model == "rg":
        parameters = {'alpha': [0.001, 0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
                      'tol': [0.00001,0.0001, 0.0005, 0.001, 0.005,.01]}
        model_name = "Ridge Classifer"
    elif model == "lso":
        parameters = {'alpha': [0, 0.001, 0.01, .1, .3, .5,.7, 1.0],
                      'tol': [0.00001,0.0001, 0.0005, 0.001, 0.005,.01]}
        model_name = "Lasso"
    elif model == "en":
        parameters = {'alpha': [0.001, 0.01, .1, .3, .5,.7, 1.0],'l1_ratio': [.1, .3, .5,.7, 1.0],
                      'tol': [0.0001, 0.0005, 0.001, 0.005]}
        model_name = "Elastic Net"
    elif model == "lg1":
        parameters = {'C': [0.00001,0.0001, 0.001, 0.01, .1,.5, 1.0],
                      'tol': [0.0001, 0.0005, 0.001, 0.005], 'solver':["liblinear"]}
        model_name = "Logistic Regression L1"
    elif model == "lg2":
        parameters = {'C': [0.00001,0.0001, 0.001, 0.01, .1,.5, 1.0],
                      'tol': [0.0001, 0.0005, 0.001, 0.005], 'solver':["newton-cg","lbfgs", "sag"]}
        model_name = "Logistic Regression L2"
    elif model == "lge":
        parameters = {'C': [0.00001,0.0001, 0.001, 0.01, .1,.5, 1.0],#'l1_ratio': [0,.1, .2, .3,.4, .5,.6,.7,.8,.9, 1], #
                      'tol': [0.0001, 0.0005, 0.001, 0.005]}#, 'solver':["sag"]}
        model_name = "Logistic Regression ElasticNet"

    return parameters, model, model_name

def run_model(model, parameters, features, target, type_pca='kpca', n_pca_components = 7, n_best = 2, transform_features = "NO", cv = ps): #cv=StratifiedKFold(10)):
    sv = SVC(kernel = 'rbf', C = 1, probability = True)
    rf = RandomForestClassifier(random_state = 44, n_jobs = 7)
    gb = GradientBoostingClassifier(random_state=0)
    kn = KNeighborsClassifier(n_jobs = 7, p = 1, algorithm ='auto')
    mlp= MLPClassifier(learning_rate='constant', max_iter=1000)
    gp = GaussianProcessClassifier(random_state=0, n_restarts_optimizer = 10, n_jobs = 8)
    dt = DecisionTreeClassifier(random_state=0)
    gnb= GaussianNB()
    qda= QuadraticDiscriminantAnalysis(priors=None)
    nsv= NuSVC(random_state=0)
    lsv= LinearSVC(dual=False, random_state=0)
    rg = RidgeClassifier(solver="auto", random_state=0)
    lso= Lasso(warm_start=False, random_state=0)
    en = ElasticNet(warm_start=False, random_state=0)
    lg1= LogisticRegression(warm_start=False, random_state=0,n_jobs =7, penalty = 'l1')
    lg2= LogisticRegression(warm_start=False, random_state=0,n_jobs =7, penalty = 'l2')
    lge= LogisticRegression(warm_start=False, random_state=0,n_jobs =7, penalty = 'elasticnet')

    if transform_features == "YES":
        if type_pca == 'kpca':
            transform = KernelPCA(kernel = 'rbf', n_components = n_pca_components)
        transform = PCA(n_components = n_pca_components)
        selection = SelectKBest(k = n_best)
        combined_features = FeatureUnion([("type_pca", transform), ("univ_select", selection)])
        features = combined_features.fit_transform(features, target)
    if model == "rf": grid_cv = GridSearchCV(rf, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = rf
    if model == "sv": grid_cv = GridSearchCV(sv, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = sv
    if model == "gb": grid_cv = GridSearchCV(gb, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = gb
    if model == "kn": grid_cv = GridSearchCV(kn, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = kn
    if model == "mlp":grid_cv = GridSearchCV(mlp, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = mlp
    if model == "gp": grid_cv = GridSearchCV(gp, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = gp
    if model == "dt": grid_cv = GridSearchCV(dt, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = dt
    if model == "gnb":grid_cv = GridSearchCV(gnb, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = gnb
    if model == "qda":grid_cv = GridSearchCV(qda, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = qda
    if model == "nsv":grid_cv = GridSearchCV(nsv, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = nsv
    if model == "lsv":grid_cv = GridSearchCV(lsv, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = lsv
    if model == "rg": grid_cv = GridSearchCV(rg, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = rg
    if model == "lso":grid_cv = GridSearchCV(lso, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = lso
    if model == "en" :grid_cv = GridSearchCV(en, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = en
    if model == "lg1" :grid_cv = GridSearchCV(lg1, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = lg1
    if model == "lg2" :grid_cv = GridSearchCV(lg2, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = lg2
    if model == "lge" :grid_cv = GridSearchCV(lge, parameters, scoring = 'roc_auc', cv = cv, n_jobs = 7, return_train_score=True); clf = lge
    return features, grid_cv, clf
print('RUN ALL THE MODELS')

if __name__ == '__main__':
    for ml_model in ["lg1","lg2","en","lso", "rg", "sv", "kn", "mlp","gb", "lsv","qda","gnb",  "nsv","gp","lge"]:#, # "dt","rf",
        parameters, model, model_name = choose_model(model=ml_model)
        features, grid_cv, clf = run_model(model, parameters, features, target, n_pca_components=7,
                                              n_best=2, transform_features="NO", cv = ps)
        print("grid_cv.fit")
        grid_cv.fit(features, target)
        print("The best classifier from GridSearchCV is: ", grid_cv.best_estimator_)
        print("Best parameters set found on development set:")
        print()
        print(grid_cv.best_params_)

        print("Grid scores on development set:")
        print()

        means_train = grid_cv.cv_results_['mean_train_score']
        means_test = grid_cv.cv_results_['mean_test_score']
        for train, test, params in zip(means_train, means_test, grid_cv.cv_results_['params']):
            print("PCA_ASD;%r;train_auc;%0.3f;validation_auc;%0.3f;%r"
                  % (model, train, test, params))
        print()

sys.stdout.close()
sys.stdout = sys.__stdout__

# Additional search methods
# BayesSearchCV
# GASearchCV
