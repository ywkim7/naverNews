import re
import pandas as pd
import datetime
import pickle
from multiprocessing import Pool, set_start_method
from functools import partial
from tqdm import tqdm
from konlpy.tag import Mecab
from nltk.tag import pos_tag
from nltk import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split



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
    pos_tokens = []

    for token in text.split():
        pos = mecab.pos(token) if re.match(r'[ㄱ-ㅎㅏ-ㅣ가-힣]+', token) else pos_tag(word_tokenize(token))      # tokenizing 후 단어와 품사 return      
        nouns = [noun for noun, pos in pos if pos.startswith('N') & (noun not in stop_words)]                   # 불용어 제거 및 pos에서 명사만 return
        tokens.extend(nouns)                                                                                    # 명사 리스트
        pos_tokens.extend([(noun, 'Noun') for noun in nouns])                                                   # 명사, 품사 튜플 리스트

    return tokens, pos_tokens


## batch 별 Preprocess, Tokenize
def process_data_in_batches(data_label_pair):
    data, labels = data_label_pair
    results = []
    for text in data:
        preprocessed_text = preprocess(text)
        tokens, pos_tokens = tokenize(preprocessed_text, mecab)
        results.append((tokens, pos_tokens))
    return list(zip(results, labels))


## Vectorize
def word_to_vector(sentences):
    embedding_model = Word2Vec(sentences=sentences, vector_size=300, min_count=5, window=6, workers=36, sg=1)
    return embedding_model



## batch 별 Get document vectors
def get_document_vectors_in_batches(embedding_dic, sentence_label_pair):
    sentences, labels = sentence_label_pair
    document_embedding_list = []
    for sentence in sentences:
        doc2vec = None
        count = 0
        for word in sentence:
            if word in embedding_dic:
                count += 1
                if doc2vec is None:
                    doc2vec = embedding_dic[word]
                else:
                    doc2vec = doc2vec + embedding_dic[word]

        if doc2vec is not None:
            doc2vec = doc2vec / count
            document_embedding_list.append(tuple(doc2vec))

    return list(zip(document_embedding_list, labels))


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

    sentences, pos_tokens = zip(*processed_data)

    embedding_model = word_to_vector(sentences)

    key_list = embedding_model.wv.index_to_key
    wv_list = embedding_model.wv.vectors

    key_wv_dic = dict(zip(key_list, wv_list))

    func = partial(get_document_vectors_in_batches, key_wv_dic)

    with Pool(num_workers) as pool:
        embedding_results = []
        for batch_results in tqdm(pool.imap_unordered(
                func, 
                ((sentences[i * batch_size : (i + 1) * batch_size], 
                Y[i * batch_size : (i + 1) * batch_size]) for i in range(num_batches))), 
                desc="Get document vectors..", total=num_batches):
            embedding_results.extend(batch_results)

    input_vectors, categories = zip(*embedding_results)

    vector_label_dic = { vector : label for vector, label in zip(input_vectors, categories)}

    with open("dataset", "wb") as f:
        pickle.dump(vector_label_dic, f)

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