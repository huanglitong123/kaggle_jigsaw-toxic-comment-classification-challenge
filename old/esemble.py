import pandas as pd
from scipy.special import logit
from scipy.special import expit
result1 = pd.read_csv(
    './result/esemble_NNBSVM_tfv_char(1-4)2_clean2_simplest_tfv_char.csv')
result2 = pd.read_csv('./result/esemble_NBSVM_tfv_char2_simplest_tfv_char.csv')
test = pd.read_csv('./data/test.csv')
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    test[label] = result1[label] * 0.5 + result2[label] * 0.5
test.drop('comment_text', axis=1, inplace=True)
test.to_csv(
    './result/esemble_NNBSVM_tfv_char(1-4)2_clean2_simplest_tfv_char_again.csv', index=False)
