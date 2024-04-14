
import os
from pathlib import Path

import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from nrclex import NRCLex
import json
import pickle
CACHE_FILENAME = "cache_database.pkl"
def create_main_path() -> Path:
    main_path = os.path.dirname(os.path.realpath(__file__))
    return Path(main_path)


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
def summarize_text(text: str):

    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    porcentaje_min_length = 0.2  # Por ejemplo, 10%
    porcentaje_max_length = 0.3  # Por ejemplo, 50%
    # Calcular el min_length basado en el porcentaje del texto original
    min_length = max(int(len(text.split()) * porcentaje_min_length), 100)
    max_length = max(int(len(text.split()) * porcentaje_max_length), 200)

    if max_length > 1024:
        max_length = 1024
    elif min_length > max_length:
        min_length = max_length - 100
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decodificar y mostrar el resumen
    resumen = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return resumen

def read_text_file(file_path: Path):
    with open(str(file_path), "r", encoding="utf-8") as file:
        text = file.read()
    return text

def list_all_files_folder_and_subfolders(folder_path: Path):
    #obtain all the files in the folder and subfolders and return the list of files
    files = []
    for file in folder_path.rglob("*"):
        if file.is_file():
            files.append(file)
    return files


def read_json_file(file_name: str):
    main_path = create_main_path()
    json_file = main_path / file_name
    if not json_file.exists():
        data = {"texto": [], "sentimiento": []}
    else:
        with open(json_file, "r") as file:
            data = json.load(file)
    return data

def save_result_to_json(result: dict, file_name: str):
    data = read_json_file(file_name)
    data["texto"].append(result["texto"])
    data["sentimiento"].append(result["sentimiento"])
    main_path = create_main_path()
    with open(main_path / file_name, "w") as file:
        json.dump(data, file)
def clasificar_sentimiento(texto):
    # Analizar el texto con NRCLex
    emociones = NRCLex(texto).affect_frequencies

    # Obtener las frecuencias de todas las emociones
    positivo = sum(emociones.get(emocion, 0) for emocion in ['joy', 'trust', 'anticipation', 'surprise','positive'])
    negativo = sum(emociones.get(emocion, 0) for emocion in ['fear', 'anger', 'sadness', 'disgust','negative'])

    # Calcular el sentimiento
    diferencia = positivo - negativo
    if diferencia > 0.05:
        return "Positivo"
    elif diferencia < -0.05:
        return "Negativo"
    else:
        return "Neutral"


def run():
    main_path = create_main_path()
    text_files_path = main_path / "results/EN"
    text_files = list(filter(lambda x: x.suffix == ".txt", list_all_files_folder_and_subfolders(text_files_path)))
    # print(len(text_files))
    text_files_without_cache = list(filter(lambda x: x.name not in get_cache_file(), text_files))

    for file in text_files_without_cache:
        text = read_text_file(file)
        txt_transcript = summarize_text(text)
        sentiment = clasificar_sentimiento(txt_transcript)
        save_dict = {"texto": txt_transcript, "sentimiento": sentiment}
        save_result_to_json(save_dict, "base_sentimen.json")
        save_in_cache(file.name)

if __name__ == "__main__":
    # run()
    Data=read_json_file("base_sentimen.json")
    pd.DataFrame(Data).to_csv("base_sentimen.csv",index=False)