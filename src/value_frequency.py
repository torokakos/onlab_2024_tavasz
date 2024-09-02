import os
import pandas as pd

def sort_csv_by_first_column(input_file, output_file):
    # adat beolvasas
    df = pd.read_csv(input_file)

    # rendezes
    df_sorted = df.sort_values(by=df.columns[1])

    # mentes
    df_sorted.to_csv(output_file, index=False)
    print(f"CSV file sorted by the first column and saved to: {output_file}")


def calculate_relative_frequency(folder_path):
    fhr_frequency = {}
    for file_name in os.listdir(folder_path):
        #if file_name.startswith("v3_"):
        if file_name.endswith(".csv"):
            print("enter")
            file_path = os.path.join(folder_path, file_name)

            data = pd.read_csv(file_path)

            fhr_counts = data['FHR'].value_counts().to_dict()

            for fhr_value, frequency in fhr_counts.items():
                if fhr_value in fhr_frequency:
                    fhr_frequency[fhr_value] += frequency
                else:
                    fhr_frequency[fhr_value] = frequency
    
    return fhr_frequency

def onefile_calculate_relative_frequency(input_file):
    fhr_frequency = {}
    print("enter")
    data = pd.read_csv(input_file)    
    fhr_counts = data['FHR'].value_counts().to_dict()
    
    for fhr_value, frequency in fhr_counts.items():
        if fhr_value in fhr_frequency:
            fhr_frequency[fhr_value] += frequency
        else:
            fhr_frequency[fhr_value] = frequency
    
    return fhr_frequency

def save_to_csv(data, output_file):
    df = pd.DataFrame(list(data.items()), columns=['FHR', 'Relative Frequency'])
    df.to_csv(output_file, index=False)
    print(f"Relative frequency data saved to: {output_file}")


folder_path = "C:\D\Documents\onlab\generated_data\gen_denorm1.csv"
frequency = onefile_calculate_relative_frequency(folder_path)

save_to_csv(frequency, "C:\D\Documents\onlab\ydata_timegan_freq.csv")

#for fhr_value, freq in sorted(frequency.items()):
    #print(f"{fhr_value},{freq:.4f}")

#sort_csv_by_first_column("C:\D\Documents\onlab\\250freq.csv", "C:\D\Documents\onlab\\250freq.csv")