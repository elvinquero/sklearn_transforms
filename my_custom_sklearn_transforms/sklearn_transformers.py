from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class KoiScore(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.scaler =  MinMaxScaler()
    
    def fit(self, X, y=None):
        df_koi_normalizado =  X[ X['koi_score'] <= 1 ]
        df_koi_por_normalizar =  X[ X['koi_score'] > 1]
        
        self.scaler.fit(df_koi_por_normalizar['koi_score'])
        return X
    
    def transform(self, X):
        df_koi_normalizado =  X[ X['koi_score'] <= 1 ]
        df_koi_por_normalizar =  X[ X['koi_score'] > 1]
        df_koi_por_normalizar['koi_score'] = self.scaler.transform(df_koi_por_normalizar['koi_score'])
        return pd.concat([df_koi_normalizado, df_koi_por_normalizar])
        
        
        
