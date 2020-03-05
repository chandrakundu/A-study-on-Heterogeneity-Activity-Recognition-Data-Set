# All the work has been done in jupyter notebook. An additional jupyter notebook file is given seperately. 

# Import necessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# Helper functions

# Function to split training and test data
def splitTrainTest(df):
    x_data = df.drop('gt', axis=1)
    y_data = df['gt']
    data = train_test_split(x_data, y_data, test_size=0.20, 
                                                    random_state=101)
    return data

# Function to score model in interms of accuracy, precision and recall
def modelScore(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return [acc, pre, rec]

# Function to draw different ROC-curve for different fpr and tpr
def drawROC(fpr, tpr, auc, param):    
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle='--')
    plt.plot(fpr, tpr, label= param+'(AUC = {0:0.2f})'
                   ''.format(auc))
    plt.legend(loc=4)
    return plt
        
# Function to fit and evaluate the model for different parameters
def modelEvaluation(model, params, data):
    # Split the data into different training and test set
    X_train, X_test, y_train, y_test = data
    
    # Set the different parameters
    paramlist = []
    for param in params: 
        for val in params[param]:
            var = {}
            var[param] = val
            paramlist.append(var)
    maxauc = 0
    bestmodel = ""
    for param in paramlist:
        
        #Fit the model
        model.set_params(**param)
        model.fit(X_train, y_train)
        
        # Model prediction
        y_pred = model.predict(X_test)
        scores = metrics.classification_report(y_test, y_pred)
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        
        # Find fpr, tpr and auc
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        
        # Find maximum AUC
        if (auc > maxauc):
            maxauc = auc 
            bestparam = param
        
        
        # Draw ROC curve
        plt = drawROC(fpr, tpr, auc, str(param)) 
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        title = type(model).__name__ + " ROC curve"
        plt.figtext(0.5, 0.01, title, wrap=True, horizontalalignment='center', fontsize=12)
    return (bestparam, plt)          


# Function to compare different model using ROC
def modelComparision(models, data):
    # Split the data into different training and test set
    X_train, X_test, y_train, y_test = data
    
    maxauc = 0
    for model in models:
        modelName = type(model).__name__ 
        
        #Fit the model        
        model.fit(X_train, y_train)
        
        # Model prediction
        y_pred = model.predict(X_test)
        scores = metrics.classification_report(y_test, y_pred)
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        
        # Find fpr, tpr and auc
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        
        # Find maximum AUC
        if (auc > maxauc):
            maxauc = auc 
            bestmodel = modelName
            bestscore = scores        

        
        # Draw ROC curve
        plt = drawROC(fpr, tpr, auc, modelName) 
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        title = "ROC Comparison"
        plt.figtext(0.5, 0.01, title, wrap=True, horizontalalignment='center', fontsize=12)
    print("Best Model: " + bestmodel)
    return (bestmodel, bestscore, plt)          



	
# Data url
phone_acc = "data/Phones_accelerometer.csv"
phone_gyro = "data/Phones_gyroscope.csv"
watch_acc = "data/Watch_accelerometer.csv"
watch_gyro = "data/Watch_gyroscope.csv"


# Read Data as string for all columns
df1 = pd.read_csv(phone_acc)
df2 = pd.read_csv(phone_gyro)
df3 = pd.read_csv(watch_acc)
df4 = pd.read_csv(watch_gyro)

# Data Merge 
drop_col = ["Index", "Arrival_Time", "Creation_Time", 'User', 'Model', 'Device']
new_df1 = df1.drop(drop_col, axis=1)
new_df2 = df2.drop(drop_col, axis=1, )
new_df3 = df3.drop(drop_col, axis=1)
new_df4 = df4.drop(drop_col, axis=1)
new_df1["sensor"] = "phone_acc"
new_df2["sensor"] = "phone_gyro"
new_df3["sensor"] = "watch_acc"
new_df4["sensor"] = "watch_gyro"

merged_df = pd.concat([new_df1, new_df2, new_df3, new_df4])


# Preprocessing Data 
df_processed = merged_df.dropna()

df = df_processed
print("Processed Data Example: ")
print(df.head(10))


# Taking sample 
df = df.sample(100000, random_state = 11)



# Factorize all categorical/string columns 

# all numerical columns
num_col = ["x", "y", "z"]
# all string columns
str_col = [x for x in df.columns if (x not in num_col)]

for col in str_col:
    # Factorize the values 
    labels, levels = pd.factorize(df[col])
    # Save the encoded variables in `iris.Class`
    df[col] = labels



# Check correlation among the features
corr = df[['x', 'y', 'z', 'sensor']].corr()
print(corr)



# Cluster analysis for reduced class data

# Build the PCA model
pca = PCA(n_components=2)

# Reduce the data, output is ndarray
reduced_data = pca.fit_transform(df)
X_pca = reduced_data[:,0]
y_pca = reduced_data[:,1]



# Draw cluster
fig, ax = plt.subplots()
ax.scatter(X_pca, y_pca, c=df["gt"], cmap="plasma", alpha=0.3)
plt.show()



# Draw cluster using seaborn
sns.scatterplot(x=X_pca, y=y_pca, hue=df["gt"])
plt.show()


# Reduce the number of class. 
# Combine sitting and standing
df.loc[df['gt'] == 0, 'gt'] = 0
df.loc[df['gt'] == 1, 'gt'] = 0
# Combine walking and stair up and down
df.loc[df['gt'] == 2, 'gt'] = 1
df.loc[df['gt'] == 3, 'gt'] = 1
df.loc[df['gt'] == 4, 'gt'] = 1
df.loc[df['gt'] == 5, 'gt'] = 1




# Cluster analysis for reduced class data

# Build the PCA model
pca = PCA(n_components=2)

# Reduce the data, output is ndarray
reduced_data = pca.fit_transform(df)
X_pca = reduced_data[:,0]
y_pca = reduced_data[:,1]



# Draw cluster
fig, ax = plt.subplots()
ax.scatter(X_pca, y_pca, c=df["gt"], cmap="plasma", alpha=0.3)
plt.show()



# Draw cluster using seaborn
sns.scatterplot(x=X_pca, y=y_pca, hue=df["gt"])
plt.show()



# Splitting training and test data
X_train, X_test, y_train, y_test = splitTrainTest(df)



# fit Decision Tree model
from sklearn import tree
param = {"criterion": ["gini", "entropy"], "max_depth": [1, 2]}
treeparam, treeplt = modelEvaluation(tree.DecisionTreeClassifier(), dtparam, splitTrainTest(df))
treemodel = tree.DecisionTreeClassifier(**treeparam)
treemodel.fit(X_train,y_train)
treepred = treemodel.predict(X_test)
treescore = modelScore(y_test, treepred)




# Random Forest 
from sklearn.ensemble import RandomForestClassifier
param = {"n_estimators": [10, 15, 20, 25]}
rfparam, plt = modelEvaluation(RandomForestClassifier(), param, splitTrainTest(df))
rfmodel = RandomForestClassifier(**rfparam)
rfmodel.fit(X_train, y_train)
rfscore = modelScore(y_test, rfmodel.predict(X_test))


# K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
param = {"algorithm": ["ball_tree", "kd_tree"], "p": [1, 2]}
knnparam, plt = modelEvaluation(KNeighborsClassifier(2), param, splitTrainTest(df))
knnmodel = KNeighborsClassifier(**knnparam)
knnmodel.fit(X_train, y_train)
knnscore = modelScore(y_test, knnmodel.predict(X_test))


# SVM
from sklearn.svm import SVC
df_svm = df.sample(10000, random_state=11)
X_train, X_test, y_train, y_test = splitTrainTest(df_svm)

param = {"C": [1, 2]}
svmparam, plt = modelEvaluation(SVC(probability=True, gamma='auto'), param, splitTrainTest(df))
svm = SVC(**svmparam)
svm.fit(X_train,y_train)
svmpred = svm.predict(X_test)
svmscore = modelScore(y_test, svmpred)


# Score table
indx = ["Accuracy", "Precision", "Recall"]
cols = ["DecisionTree", "RandomForest", "KNeighbors", "SVM"] 
score_df = pd.DataFrame(list(zip(treescore, rfscore, knnscore, svmscore)), indx, cols)
score_df


# Draw bargraph
n_classifiers = 4
accuracy = score_df.iloc[0,:]
precision = score_df.iloc[1,:]
recall = score_df.iloc[2,:]

fig, ax = plt.subplots()
idx = np.arange(n_classifiers)
bw = 0.25

plt.bar(idx, accuracy, bw, color='r', label='Accuracy')
plt.bar(idx+bw, precision, bw, color='g', label='Precision')
plt.bar(idx+bw+bw, recall, bw, color='y', label='Recall')


plt.xlabel('Classifier')
plt.ylabel('Scores')
plt.title('Scores by classifier')
plt.xticks(idx + bw, score_df.columns)
plt.legend()

plt.tight_layout()
plt.show()


# ROC for different classifiers
dtparam = {"criterion": ["gini", "entropy"], "max_depth": [1, 2]}
modelEvaluation(treemodel, dtparam, splitTrainTest(df))

# model comparision
svm = SVC(probability=True)
mm = [treemodel, rfmodel, knnmodel, svm]
modelComparision(mm, splitTrainTest(df))





