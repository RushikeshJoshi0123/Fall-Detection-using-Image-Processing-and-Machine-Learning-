#------------Importing Required Libraries--------------------------
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score, classification_report, confusion_matrix , accuracy_score, precision_score, recall_score, f1_score, roc_curve ,roc_auc_score,plot_confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

#-----------------------------------------------------------------------

#------------read the data from FinalF csv file -------------------
A = pd.read_csv("FinalF.csv",header=None)
A

df=A

df

rows,columns=df.shape

df.shape

df.head()

df.tail()

#----------Check Null values -------------------
df.isnull().values.any()

#--------Dropping 3rd column----------------
X = df.drop(columns= 3, axis=1)

Y = df[3]

X

Y
#---------------train and test Splitting -----------------
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,
                                                 test_size=0.30, random_state = 0)
# describes info about train and test set
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", Y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", Y_test.shape)



#-----------------------------------------------------------
#------------Applying Random Forest Classifier --------------
modelRF = RandomForestClassifier()
modelRF.fit(X_train,Y_train)

## produce a confusion matrix ##
plot_confusion_matrix(modelRF, X_test, Y_test)  
plt.show() 

# accuracy on training data
X_train_prediction = modelRF.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction ,Y_train)
print("--------------------------------------------------------------------------------")
print("Applying Random Forest")

# accuracy on test data
X_test_prediction = modelRF.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("--------------------------------------------------------------------------------")
print('Accuracy on Test data : ', test_data_accuracy)

print("--------------------------------------------------------------------------------")

print('Presion :',precision_score(X_test_prediction,Y_test))
print("--------------------------------------------------------------------------------")

print('Recall :',recall_score(X_test_prediction,Y_test))
print("--------------------------------------------------------------------------------")

print('F1 score : ', f1_score(X_test_prediction,Y_test))
print("-----------------------------------------------------------------------------------")
## produce a ROC plot ##
#obtain prediction probabilities
y_prob = modelRF.predict_proba(X_test)
#calculate false & true positive rates
fpr,tpr,_ = roc_curve(Y_test, y_prob[:,1])
#construct plot
plt.plot(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".8"), plt.plot([1, 1] , c=".8")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#-----------------------------------------------------------
#------------Applying Decision Tree Classifier --------------

modelDT = DecisionTreeClassifier(max_depth=9)
modelDT.fit(X_train,Y_train)

## produce a confusion matrix ##
plot_confusion_matrix(modelDT, X_test, Y_test)  
plt.show() 

# accuracy on training data
X_train_prediction = modelDT.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction ,Y_train)
print("--------------------------------------------------------------------------------")
print("Decision Tree")
print("--------------------------------------------------------------------------------")
print('Accuracy on Train data : ', train_data_accuracy)
print("--------------------------------------------------------------------------------")
# accuracy on test data
X_test_prediction = modelRF.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)
print("--------------------------------------------------------------------------------")

print('Presion :',precision_score(X_train_prediction,Y_train))
print("--------------------------------------------------------------------------------")

print('Recall :',recall_score(X_train_prediction,Y_train))
print("--------------------------------------------------------------------------------")

print('F1 score : ', f1_score(X_train_prediction,Y_train))
print("--------------------------------------------------------------------------------")


## produce a ROC plot ##
#obtain prediction probabilities
y_prob = modelRF.predict_proba(X_test)
#calculate false & true positive rates
fpr,tpr,_ = roc_curve(Y_test, y_prob[:,1])
##construct plot
plt.plot(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".8"), plt.plot([1, 1] , c=".8")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#-----------------------------------------------------------
#------------Applying KNN  Classifier --------------


modelknn=KNeighborsClassifier()
modelknn.fit(X_train, Y_train)

plot_confusion_matrix(modelknn, X_test, Y_test)
plt.show()
# accuracy on training data
X_train_prediction = modelknn.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction ,Y_train)
print("--------------------------------------------------------------------------------")
print('\nResults obtained for the knn')
print("--------------------------------------------------------------------------------")
print('\nResults obtained on Training Data')
print("--------------------------------------------------------------------------------")
print('Accuracy on Train data : ', train_data_accuracy)
print("--------------------------------------------------------------------------------")
print('Presion :',precision_score(X_train_prediction,Y_train))
print("--------------------------------------------------------------------------------")
print('Recall :',recall_score(X_train_prediction,Y_train))
print("--------------------------------------------------------------------------------")
print('F1 score : ', f1_score(X_train_prediction,Y_train))
print("--------------------------------------------------------------------------------")
# Accuracy on test data
X_test_prediction = modelknn.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("--------------------------------------------------------------------------------")

print('\nResults obtained on Testing Data')
print("--------------------------------------------------------------------------------")
print('Accuracy on Test data : ', test_data_accuracy)
print("--------------------------------------------------------------------------------")
print('Presion :',precision_score(X_test_prediction,Y_test))
print("--------------------------------------------------------------------------------")
print('Recall :',recall_score(X_test_prediction,Y_test))
print("--------------------------------------------------------------------------------")
print('F1 score : ', f1_score(X_test_prediction,Y_test))
print("--------------------------------------------------------------------------------")
print('\n')

#Calculating ROC Curve
## produce a ROC plot ##
#obtain prediction probabilities
y_prob = modelRF.predict_proba(X_test)
#calculate false & true positive rates
fpr,tpr,_ = roc_curve(Y_test, y_prob[:,1])
#construct plot
plt.plot(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".8"), plt.plot([1, 1] , c=".8")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#----------------------------------------------------------------------------------------------
#---------------------------Test the model using Single image ----------------------------------
path = '---Path---'

input_image = cv2.imread(path)
plt.imshow(input_image)

data=[]
   
#resize image
resize=(100,100)
img=cv2.resize(input_image,resize)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Initiate FAST detector
star = cv2.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
keypoints, descriptors = brief.compute(gray, kp)

out=pd.DataFrame(descriptors)
print('descriptor shape: ', out.shape)
out = out.astype(np.uint8)
hist=np.histogram(input_image,bins=[0,1,2,3])
#append the dataframe into the array 
data.append(hist[0])

#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
 

#--------As Random Forest algorithm gives highest accuracy so have tested the sample image using Random Forest Model
pred = modelRF.predict(Output)

print(pred)

if pred == 0:
    print("----------------------------------")
    print('This is  a fall image ')
    print("----------------------------------")

else:
    print("----------------------------------")
    print('This is no Fall Image')
    print("----------------------------------")
