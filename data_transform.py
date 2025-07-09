##feature engineering , data cleaning
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from source.exception import customException
from source.logger import logging
from source.utilis import save_object

@dataclass
class DataTransformationConfig:
    obj_filepath = os.path.join('artifact',"preprocessor.pk1")


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import NotFittedError

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data(self):
        numerical_columns = ["writing_score", "reading_score"]
        categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),  # Ensure scaling works with sparse matrices
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ],
            remainder='passthrough'
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data()
            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train = train_df.drop(columns=[target_column], axis=1)
            input_feature_test = test_df.drop(columns=[target_column], axis=1)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)

            train_arr = input_feature_train_arr
            test_arr = input_feature_test_arr

            print("Saved preprocessing obj")
            save_object(file_path=self.data_transformation_config.obj_filepath, obj=preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.obj_filepath

        except Exception as e:
            raise customException(e, sys)
