import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
import datetime
from transformers import BertTokenizer, BertModel
from transformers import logging
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
model.cuda()
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

def get_sentence_vector(tokens):
    sentence_embedding_list = []

    for token in tokens:
        token_tensor = torch.tensor([token]).to(device)
        segment = get_segment_id(token)
        segment_tensor = torch.tensor([segment]).to(device)
        with torch.no_grad():
            outputs = model(token_tensor, segment_tensor)
            hidden_states = outputs[2]

        # token_embeddings = torch.stack(hidden_states, dim=0)
        # token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs = hidden_states[-2][0]

        sentence_embedding = torch.mean(token_vecs, dim=0)

        sentence_embedding_list.append((sentence_embedding.cpu()).numpy())
    
    return sentence_embedding_list

def get_document_vectors_in_batches(sentences):
    document_embedding_list = []
    for sentence in sentences:
        doc2vec = None
        count = 0
        for word in sentence:
            count += 1
            if doc2vec is None:
                doc2vec = word
            else:
                doc2vec = doc2vec + word

        if doc2vec is not None:
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)

    return document_embedding_list

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
        sentence_vectors = []
        for batch_results in tqdm(pool.imap(
                get_sentence_vector, 
                ([tokens[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)])), 
                desc="Get sentence vectors..", total=num_batches):
            sentence_vectors.extend(batch_results)

    # with Pool(num_workers) as pool:
    #     doc2vecs = []
    #     for batch_results in tqdm(pool.imap(
    #             get_document_vectors_in_batches, 
    #             (sentence_vectors[i * batch_size : (i + 1) * batch_size] for i in range(num_batches))), 
    #             desc="Get document vectors..", total=num_batches):
    #         doc2vecs.extend(batch_results)


    X_train, X_test, y_train, y_test = train_test_split(np.array(sentence_vectors), np.array(categories), random_state=42, test_size=0.2)

    model = OneVsOneClassifier(SVC())

    model.fit(X_train, y_train)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    print(classification_report(y_train, train_predict))
    print(classification_report(y_test, test_predict))
    
    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time

    print("time : {}s".format(elapsed_time.total_seconds()))



if __name__ == "__main__":

    start_time = datetime.datetime.now()

    set_start_method('spawn')
    df = pd.read_csv('/app/navernews_220201_220210.csv')
    X = df['contents'].to_numpy()  # 데이터를 메모리에 미리 로드
    Y = df['category'].to_numpy()  # 데이터를 메모리에 미리 로드

    num_workers = 4
    batch_size = 16

    main(num_workers, batch_size, X, Y)