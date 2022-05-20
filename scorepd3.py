import json
import time
import sys
import os
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
import onnx
import onnxruntime 
import azureml.automl.core
from azureml.core.model import Model
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY
from azureml.core import Workspace
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType


input_sample = pd.DataFrame({"Age": pd.Series([0.0], dtype="float64"), "Sex": pd.Series([0.0], dtype="float64"), "Steroid": pd.Series([0.0], dtype="float64"), "Antivirals": pd.Series([0.0], dtype="float64"), "Fatigue": pd.Series([0.0], dtype="float64"), "Malaise": pd.Series([0.0], dtype="float64"), "Anorexia": pd.Series([0.0], dtype="float64"), "Liver Big": pd.Series([0.0], dtype="float64"), "Liver Firm": pd.Series([0.0], dtype="float64"), "Spleen Palpable": pd.Series([0.0], dtype="float64"), "Spiders": pd.Series([0.0], dtype="float64"), "Ascites": pd.Series([0.0], dtype="float64"), "Varices": pd.Series([0.0], dtype="float64"), "Bilirubin": pd.Series([0.0], dtype="float64"), "Alk Phosphate": pd.Series([0.0], dtype="float64"), "Sgot": pd.Series([0.0], dtype="float64"), "Albumin": pd.Series([0.0], dtype="float64"), "Protime": pd.Series([0.0], dtype="float64"), "Histology": pd.Series([0.0], dtype="float64")})


def init():
    global model
    model_name = 'random_forest.onnx'
    ws= Workspace.from_config()
    model_obj = Model(ws, model_name )
    model_path = model_obj.download(exist_ok = True)
    model = onnxruntime.InferenceSession(model_path)
    print(model)
@input_schema('data', PandasParameterType(input_sample))
def run(data):
    try:
        start = time.time()   # start timer
        input_data = data.values.astype(np.float32)
        input_name = model.get_inputs()[0].name  # get the id of the first input of the model   
        result = model.run([], {input_name: input_data})
        end = time.time()     # stop timer
        return {"result": np.array(result).tolist(),
                "time": end - start}
    except Exception as e:
        result = str(e)
        return {"error": result}
