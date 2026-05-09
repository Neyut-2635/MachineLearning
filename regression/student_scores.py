import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# from ydata_profiling import ProfileReport

data = pd.read_csv("../datasets/StudentScore.xls")
# profile = ProfileReport(data, title="Student Score", explorative=True)
# profile.to_file("score.html")
# print(data[["math score", "reading score", "writing score"]].corr())

target = "math score"

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# print(data["test preparation course"].unique())

# imputer = SimpleImputer(missing_values=-1, strategy='mean')
# x_train[['reading score', 'writing score']] = imputer.fit_transform(x_train[['reading score', 'writing score']])
# x_test[['reading score', 'writing score']] = imputer.transform(x_test[['reading score', 'writing score']])
#
# transform = StandardScaler()
# x_train[['reading score', 'writing score']] = transform.fit_transform(x_train[['reading score', 'writing score']])
# x_test[['reading score', 'writing score']] = transform.transform(x_test[['reading score', 'writing score']])


num_transformer = Pipeline([
    ('imputer', SimpleImputer(missing_values=-1,  strategy='median')),
    ('scaler', StandardScaler()),
])

nom_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('transform', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
])


education = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
gender = ['female', 'male']
lunch = ['standard', 'free/reduced']
test_preparation = ['none', 'completed']
ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('transform', OrdinalEncoder(categories=[education, gender, lunch, test_preparation])),
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "writing score"]),
    ("ord_feature", ordinal_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transformer, ["race/ethnicity"]),
])

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

for predict, actual in zip(y_predict, y_test):
    print(f'{predict:.1f}', actual)

