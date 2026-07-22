













































































































































































"""
Batch text-simplification utility for the DeepSeek API. It reads video descriptions directly from JSON, sorts them by video identifier, and simplifies them without relying on an intermediate text file.
"""

import json
import os
import time
from typing import List, Dict, Tuple
from openai import OpenAI



DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


MODEL_CHOICE = "deepseek-chat"


JSON_FILE_PATH = "/path/to/cinebrain/dataset/captions-qwen-2.5-vl-7b.json"
OUTPUT_FILE_PATH = "/path/to/cinebrain/dataset/captions_simplified_100token.txt"
OUTPUT_JSON_PATH = "/path/to/cinebrain/dataset/captions_simplified_100token.json"



MAX_TOKENS = 20
TEMPERATURE = 0.3






def extract_and_sort_descriptions(data: List[Dict[str, str]]) -> List[Tuple[int, str]]:
    """
    Extract video identifiers and descriptions from JSON records and return sorted pairs of video identifier and description.
    """
    sorted_descriptions = {}

    for item in data:
        video_path = item.get("video")
        text = item.get("text")

        if video_path and text:
            try:

                file_name = os.path.basename(video_path)
                video_id_str = file_name.split('.')[0]
                video_id = int(video_id_str)


                cleaned_text = text.strip()
                if cleaned_text:
                    sorted_descriptions[video_id] = cleaned_text
            except (ValueError, IndexError) as e:
                print(f"⚠️ Warning: unable to parse a video ID from path '{video_path}'; skipping. Error: {e}")
                continue


    sorted_items = sorted(sorted_descriptions.items())
    return sorted_items

def simplify_text_with_deepseek(client, original_sentence: str) -> str:
    """
    Simplify one English sentence with the DeepSeek API client and return the shortened sentence.
    """

    system_prompt = """You are a text simplification expert. Your task is to condense English sentences to be more concise while preserving core meaning. Output ONLY the simplified version without any additional text, explanations, or prefixes like 'Simplified:'."""

    user_prompt = f"""Please Simplify this sentence to no more than 120 words while keeping the main meaning. Output only the shortened version:

Original: {original_sentence}

Simplified:"""
    try:
        response = client.chat.completions.create(
            model=MODEL_CHOICE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],

            temperature=TEMPERATURE,
            stream=False
        )

        simplified_sentence = response.choices[0].message.content.strip()
        return simplified_sentence
    except Exception as e:
        print(f"⚠️ API call failed: {e}")
        return original_sentence

def append_to_txt_file(text, file_path):
    """
    Append text to a TXT file.
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        return True
    except Exception as e:
        logger.error(f"Failed to write the TXT file: {e}")
        return False

def append_to_json_file(video_id,text, file_path):
    """
    Append one entry to a JSON file.
    """

    data_list = []

    try:

        with open(file_path, 'r', encoding='utf-8') as file:

            content = file.read()
            if content:
                data_list = json.loads(content)

    except json.JSONDecodeError:

        print(f" contains invalid content and will be overwritten with new data.")
        data_list = []


    new_item = {
        "video": f"{video_id:06d}",
        "text": text
    }
    data_list.append(new_item)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to write the existing JSON file: {e}")
        return False


def main():
    """
    Run the main processing workflow.
    """

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    print("🚀 Starting the text-simplification task...")


    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print(f"📖 JSON file loaded; original record count: {len(json_data)}")
    except FileNotFoundError:
        print(f"❌ Error: JSON file not found: '{JSON_FILE_PATH}'. Check the path.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: unable to parse JSON file '{JSON_FILE_PATH}'。")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the JSON file: {e}")
        return


    sorted_items = extract_and_sort_descriptions(json_data)
    total_sentences = len(sorted_items)
    print(f"🔢 Extracted  {total_sentences}  video descriptions and sorted them by video ID.")

    if total_sentences == 0:
        print("❌ Error: no valid video descriptions were extracted.")
        return


    simplified_results = []

    for i, (video_id, sentence) in enumerate(sorted_items):
        print(f"📝 Processing sentence  {i+1}/{total_sentences}  (video ID:  {video_id:06d})...")


        simplified_sentence = simplify_text_with_deepseek(client, sentence)
        simplified_results.append((video_id, simplified_sentence))


        print(f"   Original: {sentence[:150]}{'...' if len(sentence) > 80 else ''}")
        print(f"   Simplified: {simplified_sentence}")

        append_to_txt_file(simplified_sentence, OUTPUT_FILE_PATH)
        append_to_json_file(video_id,simplified_sentence, OUTPUT_JSON_PATH)
        print(f"   ✅ Saved to TXT and JSON files.")
        print("   " + "-" * 50)
























if __name__ == "__main__":
    main()
