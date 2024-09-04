import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load and prepare the dataset
data = pd.read_csv('labeled_dysx.csv')  # Update this line to load your dataset
X = data.drop(['Label'], axis=1)
y = data['Label']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=10)

# Preprocessing the data (scaling)
sc = StandardScaler(copy=False)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define test cases for predictions
test_cases = [
    np.array([[0.5, 0.1, 0.2, 0.8, 0.3, 0.5]]),
    np.array([[0.7, 0.9, 0.4, 0.9, 0.3, 0.8]]),
    np.array([[0.1, 0.7, 0.2, 0.6, 0.9, 0.6]]),
    np.array([[0.3, 0.4, 0.5, 0.3, 0.3, 0.5]])
]

# Initialize lists to store results
models = ['DecisionTree', 'RandomForest', 'SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)','Adaboost','Adaboost(GridSearch)']
precision = []
recall = []
fscore = []
error = []
labels = np.zeros((len(models), len(test_cases)))

# Define and evaluate various models

# Model 1: Decision Tree
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
error.append(mean_absolute_error(y_test, pred_dt))
precision_dt, recall_dt, fscore_dt, _ = precision_recall_fscore_support(y_test, pred_dt, average='macro')
precision.append(precision_dt)
recall.append(recall_dt)
fscore.append(fscore_dt)

# Model 2: Random Forest
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
error.append(mean_absolute_error(y_test, pred_rf))
precision_rf, recall_rf, fscore_rf, _ = precision_recall_fscore_support(y_test, pred_rf, average='macro')
precision.append(precision_rf)
recall.append(recall_rf)
fscore.append(fscore_rf)

# Model 3: SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)
error.append(mean_absolute_error(y_test, pred_svm))
precision_svm, recall_svm, fscore_svm, _ = precision_recall_fscore_support(y_test, pred_svm, average='macro')
precision.append(precision_svm)
recall.append(recall_svm)
fscore.append(fscore_svm)

# Model 4: Random Forest with GridSearch
rf_grid_params = {'n_estimators': [10, 100, 500, 1000]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=0), rf_grid_params, scoring='f1_macro')
rf_grid.fit(X_train, y_train)
pred_rf_grid = rf_grid.predict(X_test)
error.append(mean_absolute_error(y_test, pred_rf_grid))
precision_rf_grid, recall_rf_grid, fscore_rf_grid, _ = precision_recall_fscore_support(y_test, pred_rf_grid, average='macro')
precision.append(precision_rf_grid)
recall.append(recall_rf_grid)
fscore.append(fscore_rf_grid)

# Model 5: SVM with GridSearch
svm_grid_params = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]
svm_grid = GridSearchCV(SVC(probability=True), svm_grid_params, scoring='f1_macro')
svm_grid.fit(X_train, y_train)
pred_svm_grid = svm_grid.predict(X_test)
error.append(mean_absolute_error(y_test, pred_svm_grid))
precision_svm_grid, recall_svm_grid, fscore_svm_grid, _ = precision_recall_fscore_support(y_test, pred_svm_grid, average='macro')
precision.append(precision_svm_grid)
recall.append(recall_svm_grid)
fscore.append(fscore_svm_grid)
#Model 6: adaboost
ada_model = AdaBoostClassifier(DecisionTreeClassifier(random_state=0))

ada_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'base_estimator__max_depth': [0.02,0.2,1]
}

ada_grid = GridSearchCV(ada_model, ada_params, scoring='f1_macro', cv=5)
ada_grid.fit(X_train, y_train)
pred_ada_grid = ada_grid.predict(X_test)
error.append(mean_absolute_error(y_test, pred_ada_grid))
precision_ada_grid, recall_ada_grid, fscore_ada_grid, _ = precision_recall_fscore_support(y_test, pred_ada_grid, average='macro')
precision.append(precision_ada_grid)
recall.append(recall_ada_grid)
fscore.append(fscore_ada_grid)

# Option 7: LightGBM
lgbm_model = LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=len(np.unique(y)), n_estimators=100)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)
error.append(mean_absolute_error(y_test, y_pred_lgbm))
precision_lgbm, recall_lgbm, fscore_lgbm, _ = precision_recall_fscore_support(y_test, y_pred_lgbm, average='macro')
precision.append(precision_lgbm)
recall.append(recall_lgbm)
fscore.append(fscore_lgbm)

# Compare errors of different models
print('Model\t\tError')
for i in range(len(models)):
    print(f'{models[i]}\t{round(error[i], 5)}')

plt.show()

# Confusion Matrices of different models
for i, model in enumerate([dt, rf, svm, rf_grid, svm_grid,ada_model,ada_grid,lgbm_model]):
    print(f'Confusion Matrix for {models[i]}:')
    print(confusion_matrix(y_test, model.predict(X_test)))
    # plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    print(f'For a {models[i]}: Precision = {precision[i]:.5f}, Recall = {recall[i]:.5f}, F1-score = {fscore[i]:.5f}\n')
