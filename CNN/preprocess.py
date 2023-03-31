import re
import pandas as pd
import datetime
import pickle
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from konlpy.tag import Mecab
from nltk.tag import pos_tag
from nltk import word_tokenize
from gensim.models import Word2Vec


mecab = Mecab()
STOP_WORDS_PATH = '/app/stopwords-ko.txt'
VECS_PATH = '/app/word-embeddings/word2vec/word2vec'
FILE_PATH = '/app/navernews_220201_220210.csv'


with open(STOP_WORDS_PATH) as f:
    stop_words = f.readlines()

stop_words = [stop_word.rstrip('\n') for stop_word in stop_words]

# Preprocess
def preprocess(text):
    # 한글과 영어만 남기고 특수 기호를 공백 한 칸으로 대체
    cleaned_text = re.sub(r'[^\w가-힣a-zA-Z]', ' ', text)
    return cleaned_text


# Tokenize
def tokenize(text, mecab):
    tokens = []
    for token in text.split():
        pos = mecab.pos(token) if re.match(r'[ㄱ-ㅎㅏ-ㅣ가-힣]+', token) else pos_tag(word_tokenize(token))      # tokenizing 후 단어와 품사 return      
        nouns = [noun for noun, pos in pos if pos.startswith('N') & (noun not in stop_words)]                   # 불용어 제거 및 pos에서 명사만 return
        tokens.extend(nouns)                                                                                    # 명사 리스트
    return tokens


## batch 별 Preprocess, Tokenize
def process_data_in_batches(data_label_pair):
    data, labels = data_label_pair
    results = []
    for text in data:
        preprocessed_text = preprocess(text)
        tokens = tokenize(preprocessed_text, mecab)
        results.append(tokens)
    return list(zip(results, labels))


## Vectorize
def word_to_vector(sentences):
    embedding_model = Word2Vec(sentences=sentences, vector_size=400, min_count=4, window=8, workers=36, sg=1)
    return embedding_model


def padding(sentence_list):
    data_list = []
    for sentence in sentence_list:
        while len(sentence) < 100:
            sentence.extend([0])
        if len(sentence) > 100:
            sentence = sentence[:100]
        data_list.append(tuple(sentence))

    return data_list


def main(num_workers, batch_size, X, Y):
    start_time = datetime.datetime.now()

    num_batches = (len(X) + batch_size - 1) // batch_size

    with Pool(num_workers) as pool:
        results = []
        for batch_results in tqdm(pool.imap_unordered(
                process_data_in_batches, 
                ((X[i * batch_size : (i + 1) * batch_size], 
                Y[i * batch_size : (i + 1) * batch_size]) for i in range(num_batches))), 
                desc="Tokenizing..", total=num_batches):
            results.extend(batch_results)

    processed_data, categories = zip(*results)

    embedding_model = word_to_vector(processed_data)

    with open("embedding_model", "wb") as f:
        pickle.dump(embedding_model, f)


    sentence_list = [[embedding_model.wv.key_to_index[word] for word in sentence if word in embedding_model.wv.key_to_index] for sentence in processed_data]

    data_list = padding(sentence_list)

    dataset = { sentence:label for sentence, label in zip(data_list, categories)}

    with open("dataset", "wb") as f:
        pickle.dump(dataset, f)

    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time

    print("time : {}s".format(elapsed_time.total_seconds()))



if __name__ == "__main__":

    start_time = datetime.datetime.now()

    set_start_method('spawn')
    df = pd.read_csv(FILE_PATH)
    X = df['contents'].to_numpy()  # 데이터를 메모리에 미리 로드
    Y = df['category'].to_numpy()  # 데이터를 메모리에 미리 로드

    num_workers = 36
    batch_size = 512

    main(num_workers, batch_size, X, Y)