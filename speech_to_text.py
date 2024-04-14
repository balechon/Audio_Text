
from pathlib import Path
import os
import whisper
import pickle
import pandas as pd


CACHE_FILENAME = "cache.pkl"
def list_audio_files(audio_path):
    files = list_all_files_folder_and_subfolders(audio_path)
    m4a_files = list(filter(lambda x: x.suffix == ".m4a", files))
    cache = list(get_cache_file())
    m4a_files_not_in_cache = list(filter(lambda x: x.name not in cache, m4a_files))
    return m4a_files_not_in_cache


def create_main_path() -> Path:
    main_path = os.path.dirname(os.path.realpath(__file__))
    return Path(main_path)

def list_all_files_folder_and_subfolders(folder_path: Path):
    #obtain all the files in the folder and subfolders and return the list of files
    files = []
    for file in folder_path.rglob("*"):
        if file.is_file():
            files.append(file)
    return files

def get_cache_file():
    main_path = create_main_path()
    cache_file = main_path / CACHE_FILENAME
    if cache_file.exists():
        with open(str(cache_file), "rb") as file:
            cache = pickle.load(file,encoding="utf-8")
        return cache
    else:
        return set()

def save_in_cache(file):
    cache = get_cache_file()
    cache.add(file)

    cache_file = create_main_path() / CACHE_FILENAME
    with open(str(cache_file), "wb") as file:
        pickle.dump(cache, file,protocol=pickle.HIGHEST_PROTOCOL)

def save_text_to_file(text: str, file_name: str):
    with open(str(file_name), "w",encoding="utf-8") as file:
        file.write(text)



def run():
    model = whisper.load_model("base")
    main_path = create_main_path()
    audio_path = main_path / "audio_files"

    m4a_files = list_audio_files(audio_path)

    for file in m4a_files:
        print('Transcribing file:', file.name[:10])
        result_es = model.transcribe(str(file), language="es")
        result_en = model.transcribe(str(file), language="en")
        save_text_to_file(result_es['text'], main_path /"results/ES"/ (file.name[:-4] + ".txt"))
        save_text_to_file(result_en['text'], main_path / "results/EN" / (file.name[:-4] + ".txt"))
        save_in_cache(file.name)

if __name__=="__main__":
    run()

