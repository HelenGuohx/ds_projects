
from main import *


# test data preprocess
# text_set = pd.DataFrame(test_set)
# test_set['text_len'] = test_set.question_text.apply(lambda x: len(x.split()))
# test_set["clean_text"] = test_set.question_text.apply(lambda x: text_cleaning(x))
# test_set['clean_text_len'] = test_set.clean_text.apply(lambda x: len(x.split()))
# test_set["oov_rate"] = test_set.clean_text.apply(lambda x: compute_oov_rate(x, embed_glove))
# test_set["word_vector"] = test_set.clean_text.apply(lambda x: vectorize(x, embed_glove))
# test_matrix = concate_features(test_set)
#
#
# # export files
# test_output = pd.concat(pd.DataFrame(test_matrix), test_set.target)
# test_output.to_csv('test_output.csv')

print(dataset.iloc[1])