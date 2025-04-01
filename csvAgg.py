import csv
import os

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        track_id = os.path.basename(file_path).strip(".txt")
        track_title = lines[0].strip()
        genre = ''.join(lines[-1]).strip()
        description = ''.join(lines[1:-1]).strip()[:65535]
    return track_id, track_title, description, genre

def aggregate_txt_to_csv(input_directory, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('track_id', 'track_title', 'description', 'genre'))

        for filename in os.listdir(input_directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_directory, filename)
                track_id, track_title, description, genre = process_file(file_path)
                writer.writerow([track_id, track_title, description, genre])

# Implementation
input_directory = './'
output_file = 'NewTracks.csv'
aggregate_txt_to_csv(input_directory, output_file)
