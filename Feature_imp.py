# examples of feature importance for various models

import shap
import numpy as ny

# SHAP importance of MLP
explainer = shap.KernelExplainer(model.predict, shap.sample(pd.DataFrame(x_train), 100))#, columns=feature_name))
shap_values = explainer.shap_values(pd.DataFrame(x_test), nsamples=50)
shap.summary_plot(shap_values, pd.DataFrame(x_test), plot_type="bar")
mlp_impt = np.mean(abs(shap_values[0]), axis= 0)
#alternative bar plot instead of summary plot
plt.figure()
plt.bar([x for x in range(len(mlp_impt))] ,mlp_impt)
plt.show()
# convert to original feature importance if input features are PCA components
com_tr = np.transpose(pca.components_) #155X35
MLP_IMPT = np.dot(abs(com_tr), mlp_impt) #155X35 * 35x1 = 155x1

## Other model feature imp
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression

rf  = RandomForestClassifier(random_state = 44, max_depth=3, max_features=.3, n_estimators=50, n_jobs = 7)
rf.fit(train_features, target)
rf_impt = rf.feature_importances_

et = ExtraTreesClassifier(random_state=44, max_depth=2, max_features= .3, n_estimators=50, n_jobs=7)
et.fit(train_features, target)
et_impt = et.feature_importances_ #35x1

kn = KNeighborsClassifier(n_jobs = 7, p = 1, algorithm ='auto', n_neighbors=500, leaf_size=100)
kn.fit(train_features, target)
from sklearn.inspection import permutation_importance
results = permutation_importance(kn, train_features, target, scoring='accuracy')
# get importance
kn_impt = results.importances_mean
# plot feature importance
plt.figure()
plt.bar([x for x in range(len(kn_impt))] ,kn_impt)
plt.show()

lsv= LinearSVC(dual=False, random_state=0, C = 1, loss = 'squared_hinge', penalty = 'l2',)
lsv.fit(train_features, target)
coef = lsv.coef_.reshape(-1, 1)

rg = RidgeClassifier(solver="auto", random_state=0, alpha = 0.9, tol = 0.0001)
rg.fit(train_features, target)
coef = rg.coef_.reshape(-1, 1)

lg2= LogisticRegression(warm_start=False, random_state=0,n_jobs =7, penalty = 'l2',C = .1, solver = 'newton-cg', tol = 0.001)
lg2.fit(train_features, target)
coef = lg2.coef_.reshape(-1, 1)

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=600, random_state=40, max_depth=2, learning_rate=.01, subsample=.01)
xgb.fit(train_features, target)
xgb_impt = xgb.feature_importances_

