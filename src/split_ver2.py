import os
import pandas as pd

def split_csv():
    folder_path = "D:\Documents\onlab\signals_recursive_intp"

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            # Construct the full path to the CSV file
            file_path = os.path.join(folder_path, file_name)
            out_path = os.path.join("D:\Documents\onlab\signal_seperated", file_name)
            # Load the CSV file
            data = pd.read_csv(file_path)

            # Find the index of the first non-zero value in the 'FHR' column
            first_non_zero_index = data['FHR'].ne(0).idxmax()
            
            # Initialize variables for slicing the data
            start_index = first_non_zero_index
            end_index = None
            
            # Iterate through the 'FHR' column to find zero values after the first non-zero value
            for index, value in data['FHR'].items():
                if value == 0 and start_index is not None:
                    # Found a zero value after the first non-zero value, save the data
                    end_index = index
                    output_file_path = f"{out_path.split('.')[0]}_{start_index}_{end_index}.csv"
                    data[start_index:end_index].to_csv(output_file_path, index=False)
                    print(f"Segment saved to: {output_file_path}")
                    start_index = None  # Reset start_index
                
                elif value != 0 and start_index is None:
                    # Found a non-zero value, update start_index
                    start_index = index
            
            # Check if there's remaining data after the last non-zero value
            if start_index is not None:
                output_file_path = f"{out_path.split('.')[0]}_{start_index}_end.csv"
                data[start_index:].to_csv(output_file_path, index=False)
                print(f"Segment saved to: {output_file_path}")


def delete_files_with_less_than_60_lines(folder_path):
    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            # Count the number of lines in the file
            with open(file_path, 'r') as file:
                line_count = sum(1 for line in file)
            # Delete the file if it has less than 60 lines
            if line_count < 61:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

def remove_first_column(folder_path):
    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file
            data = pd.read_csv(file_path)
            # Remove the first column
            data = data.iloc[:, 1:]
            # Save the modified DataFrame back to CSV
            data.to_csv(os.path.join("D:\Documents\onlab\signals_noseconds", file_name), index=False)
            print(f"Removed the first column from: {file_path}")


# Example usage
folder_path = "D:\Documents\onlab\signal_seperated"

remove_first_column(folder_path)

#delete_files_with_less_than_60_lines(folder_path)



# Example usage:
#split_csv()
