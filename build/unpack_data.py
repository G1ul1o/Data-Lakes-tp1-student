import os
import pandas as pad
import fnmatch

def unpack_data(input_dir, output_file):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """

    # Step 1: Initialize an empty list to store DataFrames
    
    df_list = []    

    # Step 2: Loop over files in the input directory

    for file in os.listdir(input_dir):
        
        filename = os.path.join(input_dir,file)
        print(filename)
        
        # Step 3: Check if the file is a CSV or matches a naming pattern
        if filename.endswith(".csv") or "data" in filename: 
            
            # Step 4: Read the CSV file using pandas
            df = pad.read_csv(filename)
            
            # Step 5: Append the DataFrame to the list
            df_list.append(df)
        
    # Step 6: Concatenate all DataFrames
    df_concat = pad.concat(df_list, ignore_index=True, verify_integrity=True, sort=False)
    
    # Step 7: Save the combined DataFrame to output_file
    df_concat.to_csv(output_file, index=False)
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output combined CSV file")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file)
