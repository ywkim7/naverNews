import datetime
from dataset import navernewsData
from model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def model_save(X_train, y_train):
    start_time1 = datetime.datetime.now()
    SVM = Support_Vector_Machine(X_train, y_train)

    end_time1 = datetime.datetime.now()
    elapsed_time1 = end_time1 - start_time1
    

    start_time2 = datetime.datetime.now()
    LR = Logistic_Regression(X_train, y_train)
    end_time2 = datetime.datetime.now()
    elapsed_time2 = end_time2 - start_time2
    

    start_time3 = datetime.datetime.now()
    NB = Naive_Bayes(X_train, y_train)
    end_time3 = datetime.datetime.now()
    elapsed_time3 = end_time3 - start_time3
    

    return elapsed_time1, elapsed_time2, elapsed_time3


def model_validation(X_test, y_test):
    start_time1 = datetime.datetime.now()
    trained_SVM = joblib.load('/app/PoC/ML/models/SVM.pkl')
    pred_SVM = trained_SVM.predict(X_test)
    print('-------------------------------------------\n')
    print('SVM result\n')
    print(classification_report(y_test, pred_SVM))
    print('-------------------------------------------\n')
    end_time1 = datetime.datetime.now()
    elapsed_time1 = end_time1 - start_time1
    
    start_time2 = datetime.datetime.now()
    trained_LR = joblib.load('/app/PoC/ML/models/Logistic.pkl')
    pred_LR = trained_LR.predict(X_test)
    print('-------------------------------------------\n')
    print('LR result\n')
    print(classification_report(y_test, pred_LR))
    print('-------------------------------------------\n')
    end_time2 = datetime.datetime.now()
    elapsed_time2 = end_time2 - start_time2
    
    start_time3 = datetime.datetime.now()
    trained_NB = joblib.load('/app/PoC/ML/models/Naive_Bayes.pkl')
    pred_NB = trained_NB.predict(X_test)
    print('-------------------------------------------\n')
    print('NB result\n')
    print(classification_report(y_test, pred_NB))
    print('-------------------------------------------\n')
    end_time3 = datetime.datetime.now()
    elapsed_time3 = end_time3 - start_time3

    return elapsed_time1, elapsed_time2, elapsed_time3


def main():
    dataset = navernewsData()

    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.Y, random_state=42, stratify=dataset.Y, test_size=0.2)

    # elapsed_time1, elapsed_time2, elapsed_time3 = model_save(X_train, y_train)
    # print("SVM Training time : {}s".format(elapsed_time1.total_seconds()))
    # print("Logistic Training time : {}s".format(elapsed_time2.total_seconds()))
    # print("Naive Bayes Training time : {}s".format(elapsed_time3.total_seconds()))

    elapsed_time1, elapsed_time2, elapsed_time3 = model_validation(X_test, y_test)
    print("SVM Validation time : {}s".format(elapsed_time1.total_seconds()))
    print("Logistic Validation time : {}s".format(elapsed_time2.total_seconds()))
    print("Naive Bayes Validation time : {}s".format(elapsed_time3.total_seconds()))


if __name__=='__main__':
    main()