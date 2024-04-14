
import json
import os
from pathlib import Path
from utils import get_cache_file, save_in_cache, list_all_files_folder_and_subfolders, read_text_file, save_result_to_json
from GPT import gpt

import pandas as pd
CACHE_FILENAME = "cache_openai_resume.pkl"

def create_main_path() -> Path:
    main_path = os.path.dirname(os.path.realpath(__file__))
    return Path(main_path)



def summarize_text(text: str):
    model = gpt()
    prompt = f"""
        
    Condense the following text to at least 30% of its original length, ensuring that the essence and sentiment of the original speech are fully preserved. The condensed version can be longer if necessary to maintain these elements. Do not include any explanations or additional details in your response.
    
    Text: {text}
    """
    response = model.get_completions(prompt)
    return response

def get_the_sentiment_of_the_summary(summary: str):
    model = gpt()
    prompt = f"""
    Analyze the sentiment of the following text, extracted from a speech. Rate the sentiment on a scale from 0 to 10, where 0 is completely negative and 10 is completely positive. While rating, critically assess both the positive and negative elements present in the text. Consider contradictions, challenges, or any underlying tones that might suggest a lower sentiment rating. Only the number is required.
    
    text: {summary}
    
     Do not include any explanations, additional details in your response.
    
    """
    response = model.get_completions(prompt)
    return response

def run():

    main_path = create_main_path()
    # debes especificar el path de los archivos de texto
    text_files_path = main_path / "results/EN"
    text_files = list(filter(lambda x: x.suffix == ".txt", list_all_files_folder_and_subfolders(text_files_path)))
    # print(len(text_files))
    text_files_without_cache = list(filter(lambda x: x.name not in get_cache_file(), text_files))

    for file in text_files_without_cache:
        text = read_text_file(file)
        if len(text)+100 >= 16385:
            continue
        # summary_text = summarize_text(text)
        summary_text = summarize_text(text)
        sentimen_summary = get_the_sentiment_of_the_summary(text)
        save_dict = {"summary": summary_text ,"sentiment": sentimen_summary}
        save_result_to_json(save_dict, main_path /"base_sentimen.json")
        save_in_cache(file.name, main_path / CACHE_FILENAME)



if __name__ == "__main__":
    run()