import os
import pandas as pd

# eleresi ut
folder_path = "D:\Documents\onlab\sign_intpp"
# egymas utani nullak szama
threshold = 16

def recursive_halfing_interpolation(start_idx, end_idx, fhr_values):
    try:
        num_zeros = end_idx - start_idx - 1 #nullak szama a szelso indexe kulonbsege
        if num_zeros >= 1: 
            #a kozepso index az indexek atlaga
            middle_idx = (start_idx + end_idx) // 2 
            if fhr_values[middle_idx] == 0: #ha nulla a kozepso
                left_adjacent = fhr_values[start_idx] #bal oldai ertek
                right_adjacent = fhr_values[end_idx] #jobb oldali ertek
                middle_value = (left_adjacent + right_adjacent) / 2  #a ket ertek atlaga
                fhr_values[middle_idx] = middle_value   #lesz a kozepso ertek
            
            #meghívja kozeptol balra es jobbra
            recursive_halfing_interpolation(start_idx, middle_idx, fhr_values)
            recursive_halfing_interpolation(middle_idx, end_idx, fhr_values)
            
    except IndexError:
                pass
    
#minden fajlon vegigmegy
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        #a teljes eleresi ut
        file_path = os.path.join(folder_path, file_name) 

        data = pd.read_csv(file_path) #csv betoltese
        
        consecutive_zeros_count = 0 #nulla szamlalo
         
        start_idx = None #elso nulla indexe

        for i in range(len(data)):
            #ha az ertek nulla noveli a szamlalot
            if data['FHR'][i] == 0:
                consecutive_zeros_count += 1 
                if start_idx is None: #ha meg nem volt indulo index
                    start_idx = i     #beallitja a jelenlegit
            elif start_idx is not None:
                if consecutive_zeros_count <= threshold: #thresholdon belul van
                    #meghivja a fuggvenyt a ket szelso ertek szomszedjara
                    recursive_halfing_interpolation(start_idx-1, i+1, data['FHR'].values)
                start_idx = None #visszaallitja a szamlalokat
                consecutive_zeros_count = 0

        interpolated_file_path = os.path.join("D:\Documents\onlab\signals_recursive_intp", "2_" + file_name)
        data.to_csv(interpolated_file_path, index=False)
        print(f"Interpolated data saved to: {interpolated_file_path}")
