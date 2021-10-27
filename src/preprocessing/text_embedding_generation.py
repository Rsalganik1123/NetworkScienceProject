import pickle
from tqdm import tqdm
from src.utils.utils import chunks
import argparse
from sentence_transformers import SentenceTransformer


def generate_text_features(data_list, text_key, output_key='text_emb'):
    """

    :param data_list: list of dictionary with text key
    :param text_key: key name of the text entry to compute the embedding
    :param output_key:  key name to save the embedding
    :return: updated data list
    """

    model = SentenceTransformer('all-mpnet-base-v2')

    batches = list(chunks(data_list, 32))
    for batch in tqdm(batches):
        sentences = [x[text_key] for x in batch]
        sentence_embeddings = model.encode(sentences)
        for data, emb in zip(batch, sentence_embeddings):
            data[output_key] = emb

    return data_list


def generate_text_features_file(input_path, output_path, text_key, output_key):
    """

    :param input_path: list of dictionary with text key
    :param output_path: list of dictionary with text key
    :param text_key: key name of the text entry to compute the embedding
    :param output_key:  key name to save the embedding
    :return:
    """

    print('load data')
    data_list = pickle.load(open(input_path, 'rb'))

    print('generate embedding')
    data_list = generate_text_features(data_list, text_key, output_key)

    print('Save data')
    pickle.dump(data_list, open(output_path, 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="input data path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output data path"
    )
    parser.add_argument(
        "--text_key",
        type=str,
        help="key name of the text entry to compute the embedding"
    )
    parser.add_argument(
        "--output_key",
        type=str,
        help="key name to save the embedding"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    generate_text_features_file(args.input_path, args.output_path, args.text_key, args.output_key)


if __name__ == '__main__':
    main()