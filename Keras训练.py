texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
import os
import chardet
TEXT_DATA_DIR = 'C:\TrainDataA'
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        print(name)
        label_id = len(labels_index)
        print(str(label_id))
        labels_index[name] = label_id
        #if label_id == 2:
        #    break
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
                        #print(labels)
                    except UnicodeDecodeError:
                        pass
                elif f_charInfo['encoding']=='GB2312':
                #s=s.decode(f_charInfo['encoding'])
                #if open(fpath,'r',encoding='gb2312')
                    try:
                        f = open(fpath,'r',encoding='GB2312')                
                    #print(f.read().strip())
                        texts.append(f.read().strip())
                        print(f.read().strip())
                        f.close()
                        labels.append(label_id)
                        #print(labels)
                    except UnicodeDecodeError:
                        pass
print('Found %s texts.' % len(texts))
#print(texts[0])
#print(labels)

######我们可以新闻样本转化为神经网络训练所用的张量。
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences)
# from keras.utils import np_utils
# labels = np_utils.to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)



# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_new = []
for i in indices:
    labels_new.append(labels[i])

nb_validation_samples = int(0.2 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels_new[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels_new[-nb_validation_samples:]
print(x_train[0])

###############读取词向量

embeddings_index = {}
f = open(os.path.join('C:\\', 'vectors.txt'),'r',encoding='utf-8')
for line in f.readlines():
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#############我们可以根据得到的字典生成上文所定义的词向量矩阵
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix)
#########将这个词向量矩阵加载到Embedding层中，注意，设置trainable=False使得这个编码层不可再训练。
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=data.shape[1],

                            trainable=False)


from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
sequence_input = Input(shape=(data.shape[1],), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 8, activation='relu')(embedded_sequences)
x = MaxPooling1D(8)(x)
x = Conv1D(128, 8, activation='relu')(x)
x = MaxPooling1D(8)(x)
x = Conv1D(128, 8, activation='relu')(x)
x = MaxPooling1D(50)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, 
		validation_data=(x_val, y_val),
        nb_epoch=2, batch_size=128)
model.save('C:/TrainDataA/mymodel.h5')