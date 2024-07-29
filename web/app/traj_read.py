import json

# Function to load and read a JSON file
def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Example usage
file_path = '/home/dynamo/Music/Matterport3DSimulator/web/app/val_unseen_shortest_agent.json'
json_data = load_json(file_path)

if json_data is not None:
    print(json_data)
else:
    print("Failed to load JSON data")
