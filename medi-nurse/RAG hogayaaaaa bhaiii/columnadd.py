import pandas as pd

# Load the existing TSV file
file_path = 'q6.tsv'  # Replace with your file path
df = pd.read_csv(file_path, sep='\t', header=None)

# Add column headers
df.columns = ['responses', 'Labels']

# Save the file with the new headers
output_file_path = 'q6new.tsv'  # Replace with the desired output file path
df.to_csv(output_file_path, sep='\t', index=False)

print(f"Headers added successfully. File saved as {output_file_path}")
