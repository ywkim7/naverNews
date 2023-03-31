import pandas as pd
import torch
import multiprocessing as mp
import datetime
from transformers import BertTokenizer, BertModel
from transformers import logging
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
logging.set_verbosity_error()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
model.to(device)
model.eval()


def tokenize(sentence_label_pair):
    sentences, label = sentence_label_pair
    tokens_list = []
    
    for sentence in sentences:
        tokenized_text = tokenizer(sentence, max_length=512, padding='max_length', truncation=True)
        input_ids = tokenized_text['input_ids']

        tokens_list.append(input_ids)

    return list(zip(tokens_list, label))

def get_segment_id(token):
    segment_id = [1] * len(token)
    return segment_id

def get_word_vectors(tokens):
    token_embeddings = []
    word_vecs = []

    for token in tokens:
        token_tensor = torch.tensor([token]).to(device)
        segment = get_segment_id(token)
        segment_tensor = torch.tensor([segment]).to(device)
        with torch.no_grad():
            outputs = model(token_tensor, segment_tensor)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings = token_embeddings.permute(1,0,2)

        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            word_vecs.append(sum_vec)
    
    return word_vecs


def main(num_workers, batch_size, X, Y):
    start_time = datetime.datetime.now()

    num_batches = (len(X) + batch_size - 1) // batch_size

    sentences = []
    for sentence in X:
        sentences.append(sentence)

    with Pool(num_workers) as pool:
        token_results = []
        for batch_token_results in tqdm(pool.imap_unordered(
                tokenize, 
                ((sentences[i * batch_size : (i + 1) * batch_size], 
                Y[i * batch_size : (i + 1) * batch_size]) for i in range(num_batches))), 
                desc="Tokenizing..", total=num_batches):
            token_results.extend(batch_token_results)

    tokens, categories = zip(*token_results)

    with Pool(num_workers) as pool:
        word_vectors = []
        for batch_results in tqdm(pool.imap(
                get_word_vectors, 
                ([tokens[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)])), 
                desc="Get word vectors..", total=num_batches):
            word_vectors.extend(batch_results)

    print(word_vectors[0])

    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time

    print("time : {}s".format(elapsed_time.total_seconds()))



if __name__ == "__main__":
    start_time = datetime.datetime.now()

    set_start_method('spawn')
    df = pd.read_csv('/app/navernews_220201_220210.csv')
    X = df['contents'].to_numpy()  # 데이터를 메모리에 미리 로드
    Y = df['category'].to_numpy()  # 데이터를 메모리에 미리 로드

    num_workers = 8
    batch_size = 32

    main(num_workers, batch_size, X, Y)