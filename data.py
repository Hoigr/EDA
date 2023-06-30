import pandas  as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report

import seaborn as sns
import pickle

from pyod.models.iforest import IForest

from scipy.stats import spearmanr, pearsonr, kendalltau


class Data:
    def __init__(self, url:str, path_model:str):
        self.url = url
        self.cor_tar = pd.DataFrame()
        self.path_model = path_model
        self.data = pd.read_csv(url)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.seed = 1024
        np.random.seed(self.seed)
        self.cod = {}
        self.col_type = {}
        self.stats_col = {}
        self.pred = pd.DataFrame()
        self.inf = {'SeriousDlqin2yrs':'Просрочка в 90 и более дней',
                    'RevolvingUtilizationOfUnsecuredLines':'Общий баланс средств',
                    'age':'Возраст заемщика',
                    'NumberOfTime30-59DaysPastDueNotWorse':'Просрочка в 30-59 дней за 2 года',
                    'DebtRatio':'Ежемесячные расходы',
                    'MonthlyIncome':'Ежемесячный доход',
                    'NumberOfOpenCreditLinesAndLoans':'Кол-во открытых кредитов',
                    'NumberOfTimes90DaysLate':'Сколько раз наблюдалась просрочка (90 и более дней)',
                    'NumberOfTime60-89DaysPastDueNotWorse':'кол-во задержаных платежей на 60-89 дней за 2 года',
                    'NumberOfDependents':'Кол-во иждивенцев на попечении',
                    'RealEstateLoansOrLines':'Закодированное количество кредиов',
                    'GroupAge':'Возрастная группа'}

    # Информация по датасету
    def stat(self):
        for i in self.data.columns:               
            if self.data[i].dtypes != object:
                self.stats_col[i] = {'min':self.data[i].min(),
                                    'max': self.data[i].max(),
                                    'mean':self.data[i].mean(),
                                    'median':self.data[i].median()}

    # получение данных по столбцам с порпущенными значениями
    def get_nan(self)-> dict:
        df = {}
        for i in self.data.columns:
            df[i] = round(100*self.data[i].isnull().sum(axis=0)/self.data.shape[0], 2)
        return df

     # Удаление или замена пропущенных значений
    def remove_nan(self, var:str, column:str='all'):
        if self.stats_col == {}:
            self.stat()
        var = var.lower()
        if column not in self.data.columns and column != 'all':
            raise ValueError('Неверное значение столбца')
        # Удаление столбца
        if var == 'drop_col':
            self.data = self.data.drop(IndexLabel=column, axis=1)
        # Удаление всех строк где есть значение NaN
        elif var == 'drop_row':
            self.data = self.data.dropna()
        # Замена на минимальное(st_min) максимальное(st_max)
        # ср. арифметическое(st_mean) или медианное значение(st_median)
        elif var == 'min':
            self.data[column].fillna(self.stats_col[column]['min'], inplace=True)
        elif var == 'max':
            self.data[column].fillna(self.stats_col[column]['max'], inplace=True)
        elif var == 'mean':
            self.data[column].fillna(self.stats_col[column]['mean'], inplace=True)
        elif var == 'median':
            self.data[column].fillna(self.stats_col[column]['median'], inplace=True)
                
        # Замена на предсказанное значениие (метод ближайших соседей)
        elif var == 'knn':
            self.col_type = {i:str(self.data[i].dtypes) for i in self.data.columns}
            imputer = KNNImputer(n_neighbors=3, weights='uniform')
            imputer.fit(self.data)
            imp_data = imputer.fit_transform(self.data)
            for ind, col in enumerate(self.data.columns):
                self.data[col] = imp_data[:, ind]

    def load_stat(self, url:str):
        self.stats_col = pd.read_csv(url).to_dict(orient='dict')

    # Разбиение на тренировочную и тестовую выборку
    def split_df(self, clear:bool=True):
        self.data = self.data[[i for i in self.data.columns if self.data[i].dtype!=object]]
        col = [i for i in self.data.columns if i != 'SeriousDlqin2yrs']
        if clear == False:
            x = self.data[col]

        else:
            if ''.join(self.data.columns).count('out') == 0:
                self.anomal()
            
            self.data = self.data[(self.data["out_RevolvingUtilizationOfUnsecuredLines"]==0) & \
                                    (self.data["out_age"]==0) & \
                                    (self.data["out_NumberOfTime30-59DaysPastDueNotWorse"]==0) & \
                                    (self.data["out_DebtRatio"]==0) & \
                                    (self.data["out_MonthlyIncome"]==0) & \
                                    (self.data["out_NumberOfOpenCreditLinesAndLoans"]==0) & \
                                    (self.data["out_NumberOfTimes90DaysLate"]==0) & \
                                    (self.data["out_NumberOfTime60-89DaysPastDueNotWorse"]==0) & \
                                    (self.data["out_NumberOfDependents"]==0)]

        x = self.data[col]
        y = self.data['SeriousDlqin2yrs']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                                                test_size=0.25,
                                                                                random_state=self.seed)
        self.pred['data'] = pd.DataFrame(self.y_test)['SeriousDlqin2yrs']

    # Метрики моделей обучения
    def metric(self):
        col = [i for i in self.pred.columns if i != 'data']
        for i in col:
            print('Метрика метода', i)
            print(classification_report(self.pred['data'], self.pred[i]))

    # Поиск аномалий и выбросов
    def anomal(self):
        model = IForest()
        for i in self.data.columns:
            if i != 'SeriousDlqin2yrs':
                x = (self.data[i]).to_numpy()
                x = x.reshape(len(x), 1)
                model.fit(x)
                self.data[f'out_{i}'] = model.predict(x)

    def get_anomal(self, column:str):
        if column not in self.data.columns:
            raise ValueError('Неправильно задана колонка')
        try:
            pr = round(100*self.data[f'out_{column}'].value_counts()[1]/self.data[f'out_{column}'].shape[0], 2)
        except KeyError:
            pr = 0
        finally:
            print(f"Процентное соотношение выбросов {column}: {pr}%")
            print(self.inf[column])
            sns.boxplot(x=self.data[column])

