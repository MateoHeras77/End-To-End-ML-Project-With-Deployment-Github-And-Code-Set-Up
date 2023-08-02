import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass

# Aquí se define una clase DataTransformationConfig con un atributo preprocessor_obj_file_path, 
# que se utiliza para almacenar la ruta del archivo donde se guardará el objeto del preprocesador 
# (transformador de datos) utilizado en el proceso de Machine Learning.

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:

    # Se define una clase DataTransformation, que tiene un método __init__ (constructor) que inicializa un objeto 
    # data_transformation_config de la clase DataTransformationConfig. Esto permitirá acceder a la ruta del archivo 
    # del preprocesador para guardar o cargar objetos de preprocesamiento.

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # Esta función get_data_transformer_object es responsable de crear y configurar el preprocesador (transformador de datos). 
    # Realiza el procesamiento para las columnas numéricas y categóricas, imputando valores faltantes y escalando los datos.

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        # La función initiate_data_transformation toma las rutas de los archivos de entrenamiento y prueba, 
        # lee los datos en DataFrames y aplica el preprocesador previamente configurado para transformar 
        # los datos en matrices de características listas para el entrenamiento y prueba de modelos de Machine Learning.

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

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
        
        # El archivo "preprocessor.pkl" contiene el objeto del preprocesador 
        # que ha sido configurado y ajustado con los datos de entrenamiento, 
        # y su función es aplicar transformaciones específicas para preparar 
        # los datos para su uso en un modelo de Machine Learning.
        # No es el modelo final en sí mismo, sino una herramienta para preparar 
        # los datos antes de entrenar el modelo.