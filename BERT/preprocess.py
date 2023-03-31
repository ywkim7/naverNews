import datetime
import pandas as pd
import pickle
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from transformers import BertTokenizer

FILE_PATH = "/app/navernews_220201_220210.csv"
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


def tokenize(sentence_label_pair):
    sentences, labels = sentence_label_pair
    id_list = []
    mask_list = []
    for sentence in sentences:
        tokenized_text = tokenizer(sentence, max_length=512, padding='max_length', truncation=True)
        id_list.append(tuple(tokenized_text['input_ids']))
        mask_list.append(tuple(tokenized_text['attention_mask']))

    return list(zip(zip(id_list, mask_list), labels))


def main(X, Y):
    num_workers = 32
    batch_size = 512

    num_batches = (len(X) + batch_size - 1) // batch_size

    sentences = []
    for sentence in X:
        sentences.append(sentence)

    with Pool(num_workers) as pool:
        results = []
        for batch_results in tqdm(pool.imap_unordered(
                tokenize, 
                ((sentences[i * batch_size : (i + 1) * batch_size],
                  Y[i * batch_size : (i + 1) * batch_size]) for i in range(num_batches))), 
                desc="Tokenizing..", total=num_batches):
            results.extend(batch_results)

    inputs, labels = zip(*results)
    
    input_label = { input:label for input, label in zip(inputs, labels)}

    with open("input_label", "wb") as f:
        pickle.dump(input_label, f)


if __name__=='__main__':
    start_time = datetime.datetime.now()
    
    set_start_method('spawn')

    df = pd.read_csv(FILE_PATH)
    X = df['contents'].to_numpy()
    Y = df['category'].to_numpy()

    main(X, Y)

    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time

    print("time : {}s".format(elapsed_time.total_seconds()))