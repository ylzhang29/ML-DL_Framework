from sklearn.svm import NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

rf  = RandomForestClassifier(random_state = 44, max_depth=3, max_features=.2, n_estimators=100, n_jobs = 7)
# rf = AdaBoostClassifier(rf, n_estimators= 10, random_state=0, algorithm='SAMME', learning_rate=.01)
rf.fit(X_train, y_train)
print("Training AUC :", roc_auc_score(y_train, (rf.predict_proba(X_train)[:,1])))
print("validation AUC :", roc_auc_score(y_valid, (rf.predict_proba(X_valid)[:,1])))

et = ExtraTreesClassifier(random_state=44, max_depth=4, max_features= .3, n_estimators=150, n_jobs=7)
et = AdaBoostClassifier(et, n_estimators=12, random_state=0, algorithm='SAMME', learning_rate=.02)
et.fit(X_train, y_train)
print("Train AUC :", roc_auc_score(y_train, (et.predict_proba(X_train)[:,1])))
print("validation AUC :", roc_auc_score(y_valid, (et.predict_proba(X_valid)[:,1])))

kn = KNeighborsClassifier(n_jobs = 7, p = 1, algorithm ='auto', n_neighbors=300, leaf_size=100)
kn.fit(X_train, y_train)
print("Training AUC :", roc_auc_score(y_train, (kn.predict_proba(X_train)[:,1])))
print("validation AUC :", roc_auc_score(y_valid, (kn.predict_proba(X_valid)[:,1])))

lsv= LinearSVC(dual=False, random_state=0, C = 1, loss = 'squared_hinge', penalty = 'l2',)
# lsv = AdaBoostClassifier(lsv, n_estimators=10, random_state=0, algorithm='SAMME', learning_rate=.01)
lsv.fit(X_train, y_train)
print("Training AUC :", roc_auc_score(y_train, (lsv.decision_function(X_train))))
print("validation AUC :", roc_auc_score(y_valid, (lsv.decision_function(X_valid))))

rg = RidgeClassifier(solver="auto", random_state=0, alpha = 0.9, tol = 0.0001)
# rg = AdaBoostClassifier(rg, n_estimators=10, random_state=0, algorithm='SAMME', learning_rate=.05)
rg.fit(X_train, y_train)
print("rg- validation AUC :", roc_auc_score(y_train, (rg.decision_function(X_train))))
print("validation AUC :", roc_auc_score(y_valid, (rg.decision_function(X_valid))))

lg2= LogisticRegression(warm_start=False, random_state=0,n_jobs =7, penalty = 'l2',C = .1, solver = 'newton-cg', tol = 0.001)
# lg2 = AdaBoostClassifier(lg2, n_estimators=10, random_state=0, algorithm='SAMME', learning_rate=.05).fit(pca_X_train)
lg2.fit(X_train, y_train)
print("lg2: Train AUC :", roc_auc_score(y_train, (lg2.predict_proba(X_train)[:,1])))
print("validation AUC :", roc_auc_score(y_valid, (lg2.predict_proba(X_valid)[:,1])))

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=600, random_state=40, max_depth=2, learning_rate=.01, subsample=.1)
xgb.fit(X_train, y_train)
print("Train Roc_auc_score :", roc_auc_score(y_train, (xgb.predict_proba(X_train)[:, 1])))
print("validation AUC :", roc_auc_score(y_valid, (xgb.predict_proba(X_valid)[:,1])))

# Ensemble model
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[ ("kn", kn), ('xgb', xgb), ("et", et), ("lg", lg2),  ("rf", rf)], voting  = "soft")
eclf.fit(X_train, y_train)
print("Train AUC :", roc_auc_score(y_train, (eclf.predict_proba(X_train)[:,1])))
print("validation AUC :", roc_auc_score(y_valid, (eclf.predict_proba(X_valid)[:,1])))
