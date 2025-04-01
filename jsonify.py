import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        # Convert CSV data to a list of dictionaries
        data = []
        for row in csv_reader:
            # Remove square brackets from the last field
            if row:
                last_field = list(row.keys())[-1]
                row[last_field] = row[last_field].replace('[', '').replace(']', '')
            data.append(row)

    # Write the data to a JSON file
    with open(json_file_path, mode='w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

# Example usage
csv_file_path = 'AggTracks.csv'
json_file_path = 'tracks.json'
csv_to_json(csv_file_path, json_file_path)
