import datetime
from dataset import navernewsData
from model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def model_save(X_train, y_train):
    start_time1 = datetime.datetime.now()
    XGB = make_XGBoost_model(X_train, y_train)

    end_time1 = datetime.datetime.now()
    elapsed_time1 = end_time1 - start_time1
    
    return elapsed_time1


def model_validation(X_test, y_test):
    start_time1 = datetime.datetime.now()
    trained_XGB = joblib.load('/app/PoC/ML/models/XGB.pkl')
    pred_XGB = trained_XGB.predict(X_test)
    print('-------------------------------------------\n')
    print('XGB result\n')
    print(classification_report(y_test, pred_XGB))
    print('-------------------------------------------\n')
    end_time1 = datetime.datetime.now()
    elapsed_time1 = end_time1 - start_time1
    
    return elapsed_time1


def main():
    dataset = navernewsData()

    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.Y, random_state=42, stratify=dataset.Y, test_size=0.2)

    elapsed_time1 = model_save(X_train, y_train)
    print("XGB Training time : {}s".format(elapsed_time1.total_seconds()))

    elapsed_time2 = model_validation(X_test, y_test)
    print("XGB Validation time : {}s".format(elapsed_time2.total_seconds()))


if __name__=='__main__':
    main()