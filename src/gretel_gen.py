import math

from os import path
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml

from gretel_client import configure_session
from gretel_client.helpers import poll
from gretel_client.projects.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config

from plotly.subplots import make_subplots

# Specify your Gretel API Key
configure_session(api_key="grtub9ad1872f577fb7f6392e130b7725b4abe73b232d07b54d35296de5513dd60b3", cache="no", validate=True)

#data read
folder_path = "C:\D\Documents\onlab\_testsignals110"
file_paths = [path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(".csv")]
data_frames = [pd.read_csv(file_path) for file_path in file_paths]
df = pd.concat(data_frames, ignore_index=True)


project = create_or_get_unique_project(name="DGAN-ctg")

print(f"Follow model training at: {project.get_console_url()}")

config = read_model_config("C:\D\Documents\onlab\gretel\model.yml")
config["name"] = "dgan-ctg-data"
config["models"][0]["timeseries_dgan"]["generate"] = {"num_records": 1000}

model = project.create_model_obj(model_config=config, data_source=df)
model.submit_cloud()

poll(model)


synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic_df[0:20]
synthetic_df.to_csv("C:\D\Documents\onlab\gretel\gretel_gen3.csv")

