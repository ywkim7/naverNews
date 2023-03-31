import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class Support_Vector_Machine:
    def __init__(self, X, y):
        self.X, self.y = X, y

        model = OneVsOneClassifier(SVC())
        self.model = model

        self.model.fit(self.X, self.y)

        joblib.dump(self.model, '/app/PoC/ML/models/SVM.pkl')



class Logistic_Regression:
    def __init__(self, X, y):
        self.X, self.y = X, y

        model = LogisticRegression(multi_class='ovr', n_jobs=-1, max_iter=2000)
        self.model = model

        self.model.fit(self.X, self.y)

        joblib.dump(self.model, '/app/PoC/ML/models/Logistic.pkl')



class Naive_Bayes:
    def __init__(self, X, y):
        self.X, self.y = X, y

        model = GaussianNB()
        self.model = model

        self.model.fit(self.X, self.y)

        joblib.dump(self.model, '/app/PoC/ML/models/Naive_Bayes.pkl')


class make_XGBoost_model:
    def __init__(self, X, y):
        self.X, self.y = X, y

        model = XGBClassifier(tree_method='gpu_hist', gpu_id=2)
        self.model = model

        self.model.fit(self.X, self.y)

        joblib.dump(self.model, '/app/PoC/ML/models/XGB.pkl')

class make_CatBoost_model:
    def __init__(self, X, y):
        self.X, self.y = X, y

        model = CatBoostClassifier(task_type="GPU", devices='2:3')
        self.model = model

        self.model.fit(self.X, self.y)

        joblib.dump(self.model, '/app/PoC/ML/models/Cat.pkl')