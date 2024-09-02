# Import the necessary modules

from os import path
import os
import numpy as np
import pandas as pd
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Define model parameters
gan_args = ModelParameters(batch_size=128,
                            lr=0.001,
                             betas=(0.2, 0.9),
                             latent_dim=20,
                             gp_lambda=2,
                             pac=1)

train_args = TrainParameters(epochs=10,
                                sequence_length=52,
                                sample_length=13,
                                rounds=1,
                                measurement_cols=["FHR"])
num_col= ["FHR"]
cat_col= ["UC"]

# Read the data
folder_path = "C:\D\Documents\onlab\_testsignals110"
file_paths = [path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(".csv")]
data_frames = [pd.read_csv(file_path) for file_path in file_paths]
ctg = pd.concat(data_frames, ignore_index=True)
ctg = ctg.iloc[:-8]

print("concatenated")

# Training the TimeGAN synthesizer
if path.exists('C:\D\Documents\onlab\generated_data\doppelganger_mba'):
    synth = TimeSeriesSynthesizer.load('C:\D\Documents\onlab\generated_data\doppelganger_mba')
else:
    synth = TimeSeriesSynthesizer(modelname='doppelganger', model_parameters=gan_args)
    synth.fit(ctg, train_args, num_cols=num_col, cat_cols=cat_col)
    synth.save('C:\D\Documents\onlab\generated_data\doppelganger_mba')

# Generating new synthetic samples
for i in range(0,100):
    synth_data = synth.sample(n_samples=60)
    synth_data_flat = np.concatenate(synth_data, axis=0)
    synth_data_df = pd.DataFrame(synth_data_flat, columns=ctg.columns)
    synth_data_df.to_csv(fr"C:\D\Documents\onlab\generated_data\dgan_{i}.csv", index=False)
    print(f"{i}generated")


