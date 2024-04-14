import json
import os
from pathlib import Path
import pickle


def read_json_file(json_file_path: Path):
    if not json_file_path.exists():
        data = {"summary": [], "sentiment": []}
    else:
        with open(json_file_path, "r") as file:
            data = json.load(file)
    return data

def save_result_to_json(result: dict, file_path: Path):
    data = read_json_file(file_path)
    data["summary"].append(result["summary"])
    data["sentiment"].append(result["sentiment"])
    with open(file_path, "w") as file:
        json.dump(data, file)

def read_text_file(file_name: str):
    with open(file_name, "r",encoding="utf-8") as file:
        text = file.read()
    return text
def list_all_files_folder_and_subfolders(folder_path: Path):
    #obtain all the files in the folder and subfolders and return the list of files
    files = []
    for file in folder_path.rglob("*"):
        if file.is_file():
            files.append(file)
    return files

def get_cache_file(cache_file):
    if cache_file.exists():
        with open(str(cache_file), "rb") as file:
            cache = pickle.load(file,encoding="utf-8")
        return cache
    else:
        return set()


def save_in_cache(file,cache_file):
    cache = get_cache_file(cache_file)
    cache.add(file)
    with open(str(cache_file), "wb") as file:
        pickle.dump(cache, file,protocol=pickle.HIGHEST_PROTOCOL)
