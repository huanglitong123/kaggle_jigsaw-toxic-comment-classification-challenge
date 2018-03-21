import pandas as pd
import numpy as np
from scipy.special import logit
from scipy.special import expit
import pdb
result_sum = pd.DataFrame()
result_sum = 3*pd.read_csv('./result/NBSVM_3_17_new_train.csv')
#result_sum = result_sum + 1 * pd.read_csv('./result/GRU_drop2_submission.csv')
# result_sum = result_sum + 1 * \
#    pd.read_csv('./result/GRU_Attention_submission.csv')
# result_sum = result_sum + 2 * \
#    pd.read_csv('./result/GRU_awc2_test_submission.csv')
#result_sum = result_sum + 2*pd.read_csv('./result/GRU_submission.csv')
# result_sum = result_sum + 2 * \
#    pd.read_csv('./result/GRU2_patience1_submission2.csv')
# result_sum = result_sum + 2 * \
#    pd.read_csv('./result/GRU2_patience1_submission.csv')
#result_sum = result_sum + 2*pd.read_csv('./result/GRU2_clean_submission.csv')
#result_sum = result_sum + 2*pd.read_csv('./result/esemble_awc2_GRU.csv')
result_sum = result_sum + 4 * \
    pd.read_csv('./result/submission_20180314_GRU_840_batch_size_32.csv')
result_sum = result_sum + 3 * \
    pd.read_csv('./result/submission_20180314_fasttext.csv')
result_sum = result_sum + 3 * \
    pd.read_csv('./result/submission_20180314_GRU_840_twitter.csv')
#result_sum = result_sum + 4*pd.read_csv('./result/esemble_NBSVM_awc2_GRU.csv')
test = pd.read_csv('./data/test.csv')
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    #test[label] = result1[label] * 0.5 + result2[label] * 0.5
    #test[label] = (result1[label] + result2[label] * 2 + result3[label]*2)/5
    test[label] = result_sum[label]/13
test.drop('comment_text', axis=1, inplace=True)
test.to_csv(
    './result/esemble_NB_GRU_fasttext_twitter_new_train3.csv', index=False)
