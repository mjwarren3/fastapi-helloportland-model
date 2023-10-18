from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

def select_text_data(x):
    return x['short_description']

def select_numeric_data(x):
    return x[['isNight', 'containsPortland']]

# Custom transformers
class MusicExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame['short_description'].apply(self.contains_music_or_concert).values.reshape(-1, 1)

    @staticmethod
    def contains_music_or_concert(description):
        return 1 if any(keyword in description.lower() for keyword in ['music', 'concert']) else 0


class ComedyExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame['short_description'].apply(self.contains_comedy_or_comedian).values.reshape(-1, 1)

    @staticmethod
    def contains_comedy_or_comedian(description):
        return 1 if any(keyword in description.lower() for keyword in ['comedy', 'comedian']) else 0

# Transformers
get_text_data = FunctionTransformer(select_text_data, validate=False)
get_numeric_data = FunctionTransformer(select_numeric_data, validate=False)