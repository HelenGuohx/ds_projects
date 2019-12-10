import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score  # evaluation
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.decomposition import PCA


# from main import *
# from test import *

# PCA
def reduce_demension(X, n):
    """
    X: features matrix
    n: number of compoments or total explained ratio we want
    return:
    ev: explained variance of each component
    evr: explained variance ratio of each component
    """
    pca = PCA(n_components=n)
    newX = pca.fit_transform(X)
    print("pca.explained_variance_", pca.explained_variance_)
    print("pca.explained_variance_ratio_", pd.DataFrame(pca.explained_variance_ratio_))
    print("total variance ratio", sum(pca.explained_variance_ratio_))
    return newX


# Evaluation

# %%

def evaluate_models(y_true, y_pred):
    print(classification_report(y_test, y_pred))
    print('confusion_matrix(0,1):')
    print(confusion_matrix(y_test, y_pred))
    print('cohen_kappa_score:', cohen_kappa_score(y_test, y_pred))




kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，
def find_best_model(model, param_grid, X_train, Y_train, X_test, Y_test):
    grid_search = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, cv=kflod)
    # scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
    grid_search.fit(X_train, Y_train)  # 运行网格搜索
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.cv_results_)
    y_pred = grid_search.predict(X_test)
    evaluate_models(Y_test, y_pred)
    return y_pred


def define_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    print("finish building")
    Y_predict = model.predict(X_test)
    evaluate_models(Y_test, Y_predict)
    return Y_predict

# ml models
# model = LogisticRegression(solver='saga') #Logistic Regression
# model = KNeighborsClassifier(n_neighbors=2)
model = SVC(kernel='linear', C=10)
# model = GaussianNB()
# model = DecisionTreeClassifier()  # default=”gini”
# model = RandomForestClassifier(n_estimators=100, random_state=0)

train_output = pd.read_csv('train_output.csv')
print("train_output", train_output.shape)
test_output = pd.read_csv('test_output.csv')
print("test_output ", test_output.shape)

X_train = np.nan_to_num(train_output)
y_train = np.nan_to_num(train_output.target)

X_test = np.nan_to_num(test_output)
y_test = np.nan_to_num(test_output.target)

X_train_pca = reduce_demension(X_train, 2)

# define_model(model, X_train, y_train, X_test, y_test)
# model.fit(X_train, y_train)
# print("finish building model", model)
# y_predict = model.predict(X_test)
# print("predict")
# print(y_predict)
#
# evaluate_models(y_test, y_predict)
# print("done")






# plot histogram
# sample_train_set['text_len']
# x = sample_train_set[sample_train_set['oov_rate']<0.2]['oov_rate']
# mu = np.mean(x)
# sigma = np.std(x)
#
# num_bins = 50
#
# fig, ax = plt.subplots()
#
# # the histogram of the data
# n, bins, patches = ax.hist(x, num_bins, density=1, color='#fcb43e')
#
# # add a 'best fit' line
# # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
# #      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
# # ax.plot(bins)
# ax.set_xlabel('oov_rate')
# ax.set_ylabel('Probability density')
# # ax.set_title(r'')
#
# # Tweak spacing to prevent clipping of ylabel
# fig.tight_layout()
# plt.savefig('oov_rate.png')
# plt.show()




