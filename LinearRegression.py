import numpy as np
from sklearn import impute, linear_model, metrics, preprocessing
from sklearn.model_selection import train_test_split

X_data = np.load("X_public.npy", allow_pickle=True)
y_data = np.load("y_public.npy", allow_pickle=True)
X_eval = np.load("X_eval.npy", allow_pickle=True)


X_data_text_columns = np.array(X_data[:, :10])
X_data_numeric_columns = X_data[:, 10:]

X_eval_text_columns = np.array(X_eval[:, :10])
X_eval_numeric_columns = X_eval[:, 10:]


onehot_encoder = preprocessing.OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    categories=[np.unique(X_eval_text_columns)] * X_eval_text_columns.shape[1],
)
imputer = impute.SimpleImputer(missing_values=np.nan, strategy="mean")


encoded_text_columns = onehot_encoder.fit_transform(X_data_text_columns)
encoded_numeric_columns = imputer.fit_transform(X_data_numeric_columns)

X_eval_encoded_text_columns = onehot_encoder.fit_transform(X_eval_text_columns)
X_eval_encoded_numeric_columns = imputer.fit_transform(X_eval_numeric_columns)


X_data_transformed = np.hstack((encoded_text_columns, encoded_numeric_columns))
X_eval_transformed = np.hstack(
    (X_eval_encoded_text_columns, X_eval_encoded_numeric_columns)
)


X_train, X_test, y_train, y_test = train_test_split(
    X_data_transformed, y_data, test_size=0.25
)


lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
y_test_pred = lr.predict(X_test)

r2_test = metrics.r2_score(y_test, y_test_pred)
print(f"Score: {r2_test}")

y_eval_pred = lr.predict(X_eval_transformed)
np.save("y_prediction.npy", y_eval_pred)
print("File y_prediction.npy saved")
