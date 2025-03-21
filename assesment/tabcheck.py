# Define the function to add a tab between the sentence and the digit
def add_tab_between_sentence_and_digit(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into two parts using tab as the separator
            parts = line.strip().split('\t')
            # Check if we have exactly two parts
            if len(parts) == 2:
                # Join the sentence and digit with an additional tab in between
                modified_line = f"{parts[0]}\t{parts[1]}"
                # Write the modified line to the output file
                outfile.write(modified_line + '\n')
            else:
                print(f"Line skipped due to unexpected format: {line.strip()}")

# Example usage
input_file_path = 'q4.tsv'  # Replace with your input file path
output_file_path = 'q4tab.tsv'  # Replace with your desired output file path

add_tab_between_sentence_and_digit(input_file_path, output_file_path)
