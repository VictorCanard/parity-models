import numpy as np
import os

import tiktoken

if __name__ == '__main__':

    WIKITEXT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data/wikitext/wikitext-103-raw")

    # with open(os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.train.raw"), 'r', encoding="utf8") as data_file:
    #     raw_train_data = data_file.read()

    with open(os.path.join(WIKITEXT_DATA_PATH, "wiki.test.raw"), 'r', encoding="utf8") as data_file:
        raw_test_data = data_file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    #raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
    raw_tokenized_test = tokenizer.encode_ordinary(raw_test_data)

    #train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16)
    test_tokenized = np.array(raw_tokenized_test, dtype=np.uint16)

    #train_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, 'train.bin'))
    test_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, 'test.bin'))

    print("completed the tokenization process!")

