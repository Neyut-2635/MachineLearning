import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv('/Users/thanhtuyen2603/Documents/ML/MachineLearning/datasets/csgo.csv')

# profile = ProfileReport(df, title='CSGO', explorative=True)
# profile.to_file('csgo_report.html')

x = df.drop(['result', 'day', 'date', 'month', 'year', 'wait_time_s'], axis=1)
y = df['result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=-1, strategy='mean')),
    ('scaler', StandardScaler()),
])

norm_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
    ('scaler', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(transformers=[
    ('num_transformer', num_transformer, [
        'match_time_s',
        'team_a_rounds',
        'team_b_rounds',
        'ping',
        'kills',
        'assists',
        'deaths',
        'mvps',
        'hs_percent',
        'points'
    ]),
    ('norm_transformer', norm_transformer, [
        'map'
    ])
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logistic_regression', LogisticRegression())
])



model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))




