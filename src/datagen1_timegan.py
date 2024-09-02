# Import the necessary modules

from os import path
import os
import numpy as np
import pandas as pd
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Define model parameters
gan_args = ModelParameters(batch_size=128,
                            lr=5e-4,
                            noise_dim=8,
                            layers_dim=64,
                            latent_dim=64,
                            gamma=1)

train_args = TrainParameters(epochs=2,
                                sequence_length=60,
                                number_sequences=2)

# Read the data
folder_path = "D:\Documents\onlab\_testsignals110"
file_paths = [path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(".csv")]
data_frames = [pd.read_csv(file_path) for file_path in file_paths]
ctg = pd.concat(data_frames, ignore_index=True)

print("concatenated")

# Training the TimeGAN synthesizer
if path.exists('D:\Documents\onlab\generated_data\synthesizer110_ctg.pkl'):
    synth = TimeSeriesSynthesizer.load('D:\Documents\onlab\generated_data\synthesizer110_ctg.pkl')
else:
    synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
    synth.fit(ctg, train_args, num_cols=ctg.columns)
    synth.save('D:\Documents\onlab\generated_data\synthesizer110_ctg.pkl')

# Generating new synthetic samples
for i in range(0,100):
    synth_data = synth.sample(n_samples=10)
    synth_data_flat = np.concatenate(synth_data, axis=0)
    synth_data_df = pd.DataFrame(synth_data_flat, columns=ctg.columns)
    synth_data_df.to_csv(fr"D:\Documents\onlab\generated_data\synthetic_data_{i}.csv", index=False)
    print(f"{i}generated")


