import re

import pandas as pd

from sklearn.model_selection import train_test_split


CLEAN_DATA_PATH = "data/dataset_processed.txt"


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
NON_STANDARD_PATTERN = re.compile(r"[^a-z0-9\s]")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def load_data(path):
    """Чтение файла с сырыми данными"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    return pd.DataFrame({"text": lines})


def clean_data(text):
    """Очистка данных:
    1. Приведение к нижнему регистру
    2. Удаление ссылок, упоминаний, эмодзи
    3. Замена нестандартных символов на пробел
    4. Удаление лишних пробелов
    """
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = NON_STANDARD_PATTERN.sub(" ", text)

    # удаление лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text


def save_cleaned_data(df, out_file):
    """Сохранение очищенных данных в отдельный файл"""
    df["cleaned_text"].to_csv(
                                out_file, 
                                index=False,
                                header=False
                            )


def load_and_preprocess(path):
    """Загрузка файла и очистка данных"""
    df = load_data(path)

    df["cleaned_text"] = df['text'].apply(clean_data)
    save_cleaned_data(df, CLEAN_DATA_PATH)
    return df["cleaned_text"]
    
    
def split_data(df, random_state=42):
    """
    Разбиение соответственно условиям:
    train = 80%
    val = 10%
    test = 10%
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        shuffle=True
    )

    return train_df, val_df, test_df
