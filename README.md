# kaggle_jigsaw-toxic-comment-classification-challenge
kaggle jigsaw-toxic-comment-classification-challenge
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
初做文本比赛
尝试了LR,NB,SVM,LGB,LSTM,GRU
深度学习较LR之类的效果要好
尝试不同GRU比lstm好些
glove比fasttest好些

关键文件，GRU_fasttext.py是核心
NLP_model.py是测试模型用的

缺点：
1. 测试模型时候，没加KFOLD,结果会存在抖动，无法准确检验模型准确性
2. 在选用机器学习模型时候，缺少不断尝试的努力，只是简单的测试，然后选用最好的模型
3. 在深度学习模型中，没有建立快速调参的方法（缺乏设备）
4. 缺少记录日志
5. 缺少对模型变好变坏的分析和预测
6. 对自己新想法的实施通常比较慢
