import json
import os
from typing import List, Dict, Any






file_path = "/path/to/cinebrain/dataset/captions-qwen-2.5-vl-7b.json"
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")

    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{file_path}'.")

    exit()


output_file = '/path/to/cinebrain/dataset/captions-qwen-2.5-vl-7b.txt'



def extract_and_sort_descriptions(data: List[Dict[str, str]]) -> Dict[int, str]:
    """
    Extract video identifiers and descriptions from JSON records and sort them by identifier.
    """
    sorted_descriptions: Dict[int, str] = {}

    for item in data:
        video_path = item.get("video")
        text = item.get("text")

        if video_path and text:


            try:

                file_name = os.path.basename(video_path)

                video_id_str = file_name.split('.')[0]

                video_id = int(video_id_str)

                sorted_descriptions[video_id] = text
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse video ID from path '{video_path}'. Skipping. Error: {e}")
                continue



    sorted_items = sorted(sorted_descriptions.items())



    return sorted_items

def write_descriptions_to_txt(sorted_items: List[tuple[int, str]], output_path: str):
    """
    Write sorted video descriptions to a TXT file.
    """
    print(f"Starting to write descriptions to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for video_id, text in sorted_items:


                formatted_id = f"{video_id:06d}"



                f.write(text)
                f.write("\n")

        print(f"Successfully wrote {len(sorted_items)} descriptions to '{output_path}'.")
    except IOError as e:
        print(f"Error: Could not write to file '{output_path}'. Error: {e}")




sorted_items = extract_and_sort_descriptions(json_data)


write_descriptions_to_txt(sorted_items, output_file)