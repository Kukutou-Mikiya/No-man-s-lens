texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
import os
import chardet

TEXT_DATA_DIR = os.getcwd()
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        ##if label_id == 2:
            ##break
        for fname in sorted(os.listdir(path)):
            if 1:
            #if fname.isdigit():
                fpath = os.path.join(path, fname)
                #f = open(fpath,'r',encoding='latin-1')
                f = open(fpath,'rb')
                s=f.read()
                f_charInfo=chardet.detect(s)
                f.close()
                #print(f_charInfo['encoding'])
                if f_charInfo['encoding']=='ascii':
                #s=s.decode(f_charInfo['encoding'])
                #if open(fpath,'r',encoding='gb2312')
                    try:
                        f = open(fpath,'r',encoding='ascii')                
                    #print(f.read().strip())
                        texts.append(f.read().strip())
                        f.close()
                        labels.append(label_id)
                    except UnicodeDecodeError:
                        pass
                elif f_charInfo['encoding']=='GB2312':
                #s=s.decode(f_charInfo['encoding'])
                #if open(fpath,'r',encoding='gb2312')
                    try:
                        f = open(fpath,'r',encoding='GB2312')                
                    #print(f.read().strip())
                        texts.append(f.read().strip())
                        f.close()
                        labels.append(label_id)
                    except UnicodeDecodeError:
                        pass

print('Found %s texts.' % len(texts))

######我们可以新闻样本转化为神经网络训练所用的张量。
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences,maxlen=7746)
print('Shape of data tensor:', data.shape)

'''
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=data.shape[1],

                            trainable=False)
'''

'''from keras.models import load_model
#import keras.metrics
model = load_model('C:\TrainData\mymodel.h5')
#predict = model.predict_classes(TEXT_DATA_DIR)
prob = model.predict(data,batch_size=128,verbose=1)
pred = [float(data) for data in prob]
print(pred)
k=1
for i in predict:
    print ('文本 '+str(k)+'的预测结果为：')
    print(i)
    k+=1#'''
modelh5 = os.getcwd()+'\\mymodel.h5'    
from keras.models import *
model = load_model(modelh5)
prob = model.predict(data,batch_size=128,verbose=1)
#pred = [float(data) for data in prob]
k=0
for i in prob:
    print('第'+str(k)+'文本')
    print(i)
    k+=1