import re
import pandas as pd
from konlpy.tag import Mecab
from nltk.tag import pos_tag
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, set_start_method
import multiprocessing as mp
from tqdm import tqdm
import datetime


mecab = Mecab()

# Preprocess
def preprocess(text):
    # 한글과 영어만 남기고 특수 기호를 공백 한 칸으로 대체
    cleaned_text = re.sub(r'[^\w가-힣a-zA-Z]', ' ', text)
    return cleaned_text

# Tokenize
def tokenize(text, mecab):
    tokens = []
    pos_tokens = []

    for token in text.split():
        pos = mecab.pos(token) if re.match(r'[ㄱ-ㅎㅏ-ㅣ가-힣]+', token) else pos_tag(word_tokenize(token))      # tokenizing 후 단어와 품사 return      
        nouns = [noun for noun, pos in pos if pos.startswith('N') and len(noun) >= 2]                           # pos에서 단어 길이가 2 이상인 명사만 return
        tokens.extend(nouns)                                                                                    # 명사 리스트
        pos_tokens.extend([(noun, 'Noun') for noun in nouns])                                                   # 명사, 품사 튜플 리스트

    return tokens, pos_tokens

def process_data_in_batches(data_label_pair):
    data, labels = data_label_pair
    results = []
    for text in data:
        preprocessed_text = preprocess(text)
        tokens, pos_tokens = tokenize(preprocessed_text, mecab)
        results.append((tokens, pos_tokens))
    return list(zip(results, labels))


def main(num_workers, batch_size, X, Y):
    start_time = datetime.datetime.now()

    num_batches = (len(X) + batch_size - 1) // batch_size

    with Pool(num_workers) as pool:
        results = []
        for batch_results in tqdm(pool.imap_unordered(
                process_data_in_batches, 
                ((X[i * batch_size : (i + 1) * batch_size], 
                Y[i * batch_size : (i + 1) * batch_size]) for i in range(num_batches))), 
                desc="Tokenizer..", total=num_batches):
            results.extend(batch_results)

    processed_data, categories = zip(*results)

    tokens, pos_tokens = zip(*processed_data)

    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time

    return elapsed_time.total_seconds()


def test():
    set_start_method('spawn')
    df = pd.read_csv('/app/data/navernews_220201_220210.csv')
    X = df['contents'].to_numpy()  # 데이터를 메모리에 미리 로드
    Y = df['category'].to_numpy()  # 데이터를 메모리에 미리 로드

    # num_workers_options = [2, 6, 10, 14, 18, 22, 26, 30]
    default_worker = mp.cpu_count() // 3
    num_workers_options = [default_worker, default_worker+2, default_worker+4, default_worker+6, default_worker+8, default_worker+10]
    batch_size_options = [64, 128, 256, 512]

    best_time = float('inf')
    best_config = None

    for batch_size in batch_size_options:
        for num_workers in num_workers_options:
            elapsed_time = main(num_workers, batch_size, X, Y)
            print(f"num_workers: {num_workers}, batch_size: {batch_size}, time: {elapsed_time}s")
            if elapsed_time < best_time:
                best_time = elapsed_time
                best_config = (num_workers, batch_size)

    print(f"Best configuration: num_workers: {best_config[0]}, batch_size: {best_config[1]}, time: {best_time}s")

if __name__ == "__main__":

    start_time = datetime.datetime.now()

    set_start_method('spawn')
    df = pd.read_csv('/app/data/navernews_220201_220210.csv')
    X = df['contents'].to_numpy()  # 데이터를 메모리에 미리 로드
    Y = df['category'].to_numpy()  # 데이터를 메모리에 미리 로드

    num_workers = 36
    batch_size = 512
    num_batches = (len(X) + batch_size - 1) // batch_size

    with Pool(num_workers) as pool:
        results = []
        for batch_results in tqdm(pool.imap_unordered(
                process_data_in_batches, 
                ((X[i * batch_size : (i + 1) * batch_size], 
                Y[i * batch_size : (i + 1) * batch_size]) for i in range(num_batches))), 
                desc="Tokenizer..", total=num_batches):
            results.extend(batch_results)

    processed_data, categories = zip(*results)

    tokens, pos_tokens = zip(*processed_data)

    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time
    print("time : {}s".format(elapsed_time.total_seconds()))
