import data
from catboost import CatBoostClassifier


class Model(data.Data):
    # Градиентный бустинг
    def cat(self, save:bool=False):
        model = CatBoostClassifier(iterations=100,
                                   eval_metric='Accuracy',
                                   learning_rate=0.05,
                                   depth=5,
                                   random_seed=self.seed)

        model.fit(self.x_train,
                  self.y_train,
                  eval_set=(self.x_test, self.y_test))

        self.pred['catboost'] = model.predict(self.x_test)
        if save:
            model.save_model(self.path_model+'cat_model', format="cbm")

    def load_predict(self)-> tuple:
        df = data.pd.read_csv('pred.csv')
        url = r'Model\cat_model'
        df.drop('GroupAge', axis=1, inplace=True)
        df.drop('RealEstateLoansOrLines', axis=1, inplace=True)
        model = CatBoostClassifier()
        model.load_model(url, format='cbm')
        df['predict'] = model.predict(df)

        return (df['predict'] )