import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path = os.path.join("artifact" , 'preprocesser.pkl')

class DataTransformation :
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns = ['writing_score' , 'reading_score']

            categorical_columns = [
                'gender' ,
                "parental_level_of_education" ,
                "lunch" ,
                "test_preparation_course" ,
                'race_ethnicity'
            ]

            num_pipeline = Pipeline(
                steps=[ 
                    ("imputer" , SimpleImputer(strategy='median')) ,
                    ("Scalar" , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer" , SimpleImputer(strategy='most_frequent')) ,
                    ("one_hot_encoder" , OneHotEncoder()) ,
                    ("Scalar" , StandardScaler(with_mean=False)) 
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocesser = ColumnTransformer(
                [
                    ("num_pipeline" , num_pipeline , numerical_columns) ,
                    ('cat_pipeline' , cat_pipeline , categorical_columns)
                ]
            )

            return preprocesser
        except Exception as e:
            raise CustomException(e , sys)
    
    def initiate_data_transformation(self , train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Train and Test Data Completed.")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            # Dividng dataset into Dependent and Independent Features
            # we have to seperate features for both train and test dataset
            target_column_name = 'math_score'
            numerical_columns = ['reading_score' , 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name] , axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
   
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Combining the Indepedent and dependent features
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        