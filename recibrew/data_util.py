from nltk.tokenize import word_tokenize
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger('recibrew')


def construct_torchtext_iterator(train_csv: str, dev_csv: str, test_csv: str, device: str = 'cuda',
                                 batch_size: int = 64, max_vocab: int = 3000, fix_length=144) -> Dict[str, Any]:
    """
    Construct the iterator used to train the data.

    :param train_csv: train_csv file csv
    :param dev_csv: dev_csv file csv
    :param test_csv: test_csv file csv
    :param device: device of the torch tensor ('cpu' or 'cuda')
    :param batch_size : the batch size of each iterator (train, dev, test)
    :param max_vocab : max vocab in dictionary
    :return: Dictionary return iterators and src and tgt field (for accessing its vocabulary)
    """
    logger.info("Reading file {}, {}, {}".format(train_csv, dev_csv, test_csv))
    src_field = Field(sequential=True, tokenize=word_tokenize, lower=True, init_token='<s>', eos_token='</s>',
                      fix_length=fix_length)
    tgt_field = Field(sequential=True, tokenize=word_tokenize, lower=True, init_token='<s>', eos_token='</s>',
                      fix_length=fix_length )
    fields = [('src', src_field), ("tgt", tgt_field)]
    train_ds, dev_ds, test_ds = TabularDataset.splits(path='.', format='csv', train=train_csv, validation=dev_csv,
                                                      test=test_csv, skip_header=True,
                                                      fields=fields)
    vocab_df = pd.read_csv(train_csv)
    build_vocab_field(vocab_df, src_field, tgt_field, max_vocab)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_ds, dev_ds, test_ds),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.src),
        sort_within_batch=False,
    )
    return dict(
        train_iter=train_iter,
        val_iter=val_iter,
        test_iter=test_iter,
        src_field=src_field,
        tgt_field=tgt_field
    )


def build_vocab_field(vocab_df, src_field, tgt_field, max_vocab=5000):
    src_list = vocab_df.src.apply(word_tokenize).tolist()
    tgt_list = vocab_df.tgt.apply(word_tokenize).tolist()
    src_list.extend(tgt_list)  # src_list = src_list + tgt_list vocab
    src_field.build_vocab(src_list, max_size=max_vocab)
    tgt_field.build_vocab(src_list, max_size=max_vocab)
