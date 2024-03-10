import ollama
import pandas as pd
from tqdm import tqdm  
def break_into_sections(input_text):
    # Define markers for each section
    markers = {
        'title': '[[[Title]]]',
        'content': '[[[Content]]]',
        'user': '[[[User]]]',
        'assistant': '[[[Assistant]]]'
    }
    
    # Initialize a dictionary to hold the extracted sections
    sections = {
        'title': '',
        'content': '',
        'user': '',
        'assistant': ''
    }
    
    # Initialize a variable to keep track of the current section being processed
    current_section = None
    
    position = 0
    while position < len(input_text):
        found_marker = False
        for section, marker in markers.items():
            marker_position = input_text.find(marker, position)
            if marker_position == position:  # Marker found at the current position
                current_section = section
                position += len(marker)  # Move past the marker
                found_marker = True
                break
        if not found_marker:
            if current_section is not None:
                next_marker_position = len(input_text)
                for marker in markers.values():
                    temp_position = input_text.find(marker, position)
                    if 0 <= temp_position < next_marker_position:
                        next_marker_position = temp_position
                sections[current_section] += input_text[position:next_marker_position]
                position = next_marker_position
            else:
                position += 1
    
    for section in sections:
        sections[section] = sections[section].strip()
    
    return sections

def process_dataset(csv_file_path, output_csv_file_path):
    df = pd.read_csv(csv_file_path)
    output_data = []    
    for i, row in tqdm(enumerate(df.itertuples()), total=len(df), desc="Processing"):

        title = getattr(row, "title")
        content = getattr(row, "content")

        message = f"""[[[Title]]] {title}
[[[Content]]] {content} 
The following is an interaction between a user and an AI assistant that is related to the above text.
[[[User]]] """

        response =  ollama.generate(model='eas/nous-genstruct:7b-q8_0', prompt=message)
        
        gen_text = response['response']
        combined_message = message + "\n " + gen_text
        print(combined_message)
        sectioned = break_into_sections(combined_message)
        
        # Append the output item to the output_data list, maintaining the CSV structure
        output_data.append([title, content, sectioned['user'], sectioned['assistant']])

    # Convert the list of lists into a DataFrame
    output_df = pd.DataFrame(output_data, columns=["title", "content", "user", "assistant"])
    
    # Write the DataFrame to a CSV file
    output_df.to_csv(output_csv_file_path, index=False)

# Example usage of the function with sample file paths
csv_file_path = "sample_data.csv"  # Replace this with the actual file path
output_csv_file_path = "output_data.csv"  # Specify the output CSV file path
process_dataset(csv_file_path, output_csv_file_path)
