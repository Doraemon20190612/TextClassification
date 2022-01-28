from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def word_ngrams(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    parameter = input_['parameter']

    ###########################################

    def token_ngrams(tokens, ngram_range=(1, 1)):
        # handle token n-grams
        min_n, max_n = ngram_range
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens.append("".join(original_tokens[i: i + n]))
        return tokens

    text_part_ll = [token_ngrams(tokens=i,
                                 ngram_range=(parameter['TextPreprocess']['word_ngrams']['ngram_range'][0],
                                              parameter['TextPreprocess']['word_ngrams']['ngram_range'][1]))
                    for i in tqdm(text_part_ll)]
    text_part_sl = [' '.join(i) for i in tqdm(text_part_ll)]
    text_predict_part_ll = [token_ngrams(tokens=i,
                                         ngram_range=(parameter['TextPreprocess']['word_ngrams']['ngram_range'][0],
                                                      parameter['TextPreprocess']['word_ngrams']['ngram_range'][1]))
                            for i in tqdm(text_predict_part_ll)]
    text_predict_part_sl = [' '.join(i) for i in tqdm(text_predict_part_ll)]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    logging.info('ngram已完成')
    return output_
