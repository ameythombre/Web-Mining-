import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def classify(training_file, file_to_test_file): 
    train = pd.read_csv(training_file, header=0)
    file_to_test = pd.read_csv(file_to_test_file,header=0)

    tfidf_vect = TfidfVectorizer() 
    clf = svm.LinearSVC()
    metric = 'f1_macro'

    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', svm.LinearSVC())
                       ])

    parameters = {'tfidf__min_df' : [1,2,5] ,
                  'tfidf__stop_words' : [None,"english"] ,
                  'clf__C': [0.5,1.0,5.0],
    }

    gs_clf = GridSearchCV    (text_clf, param_grid=parameters,      scoring=metric, cv=6)


    gs_clf = gs_clf.fit(train["text"], train["label"])

    for parameter in gs_clf.best_params_:
        print(parameter,": ",gs_clf.best_params_[parameter])

    print("f1 score is:", gs_clf.best_score_)

    
    y_pred = gs_clf.predict(file_to_test["text"])

    uniqueLabels=sorted(train["label"].unique())

    precision, recall, fscore, support=         precision_recall_fscore_support(         file_to_test["label"],y_pred,labels=uniqueLabels)

    print("f-score: ", fscore)


    
def impact_of_sample_size(train_file, test_file):
    dataTrain = pd.read_csv(train_file, header=0)
    test = pd.read_csv(test_file, header=0)

    dataTrainList = dataTrain["text"].values.tolist()

    n  = round(len(dataTrainList) / 300)
    total = 0
    iterator = 0

    precisionSVM_Model = []
    recallSVM_Model = []
    precisionMNB_Model = []
    recallMNB_Model = []
    sampleList = []
    while iterator < n:
        iterator += 1
        total += 300
        print(total)
        sampleList.append(total)
        train = dataTrain[0:total]
        
        text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                             ('clf',MultinomialNB())
                           ])

        parameters = {'tfidf__stop_words' : ["english"]}

        gs_clf = GridSearchCV        (text_clf, param_grid=parameters)


        gs_clf = gs_clf.fit(train["text"], train["label"])

        y_pred = gs_clf.predict(test["text"])

        labels=sorted(train["label"].unique())


        precision_MNB, recall_MNB, fscore, support=             precision_recall_fscore_support(             test["label"],y_pred , labels=labels, average = "macro")
            
        precisionMNB_Model.append(precision_MNB *100)
        recallMNB_Model.append(recall_MNB * 100)

        text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                             ('clf',svm.LinearSVC())
                           ])

        parameters = {'tfidf__stop_words' : ["english"]}

        gs_clf = GridSearchCV        (text_clf, param_grid=parameters)

        gs_clf = gs_clf.fit(train["text"], train["label"])

        y_pred = gs_clf.predict(test["text"])

        precision_SVC, recall_SVC, fscore, support=             precision_recall_fscore_support(             test["label"], y_pred, labels=labels, average = "macro")

        
        precisionSVM_Model.append(precision_SVC * 100)
        recallSVM_Model.append(recall_SVC * 100)
    
    
    plt.plot(sampleList, precisionMNB_Model, color='black')
    plt.plot(sampleList, precisionSVM_Model, color='red')
    plt.xlabel('No. of Samples')
    plt.ylabel('Macro Precision')
    plt.title('Relationship between Sample Size and Precision')
    plt.show()

    plt.plot(sampleList, recallMNB_Model, color='black')
    plt.plot(sampleList, recallSVM_Model, color='red')
    plt.xlabel('No. of Samples')
    plt.ylabel('Macro Recall')
    plt.title('Relationship between Sample Size and Recall')
    plt.show()



if __name__ == "__main__":

    training_file_path = "news_train.csv"
    testing_file_path = "news_test.csv"
    data_set = "amazon_review_500.csv"
    classify(training_file_path, testing_file_path)
    impact_of_sample_size(training_file_path, testing_file_path)

