import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.title('MACHINE LEARNING CLASSIFIER')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine','Diabetes','boston')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest','Decision Tree')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    elif name== 'Diabetes':
        data = datasets.load_diabetes()
    elif name== 'boston':
        data = datasets.load_boston()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        kernel = st.sidebar.selectbox('Select Kernel',('rbf','linear','poly','sigmoid','precomputed'))
        params['kernel'] = kernel
        degree = st.sidebar.slider('Degree',2,5)
        params['d']=degree
        gamma = st.sidebar.selectbox('select gamma',('scale','auto'))
        params['gm']=gamma

    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
        # weights = st.sidebar.selectbox('Select Weight',('uniform','distance'))
        # params['wt']=weights
        algo = st.sidebar.selectbox('Select Algorithm',('auto','ball_tree','kd_tree','brute'))
        params['algo']=algo
        leaf_size = st.sidebar.slider('Choose Leaf size',1,80)
        params['ls']=leaf_size
        pp=st.sidebar.selectbox('Select Power Parameter',('1','2'))
        params['pp']=pp
    elif clf_name == 'Decision Tree':
        maxf = st.sidebar.selectbox('Select max_features',('auto','sqrt','log2'))
        params['mf']=maxf
        maxd= st.sidebar.slider('Choose max_depth',1,10)
        params['md']=maxd
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        n_jobs = st.sidebar.slider('Choose Number of jobs',-1,10)
        params['nj']=n_jobs
        verbos = st.sidebar.slider('Verbosity of the tree Building',0,10)
        params['vb']=verbos
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'],kernel=params['kernel'],degree=params['d'],gamma=params['gm'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'],weights=params['wt'],algorithm=params['algo'],leaf_size=params['ls'],p=params['pp'])
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(max_features=params['mf'],max_depth=params['md'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], n_jobs=params['nj'] ,random_state=1234,verbose=params['vb'])
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy ={acc*100} %')

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
