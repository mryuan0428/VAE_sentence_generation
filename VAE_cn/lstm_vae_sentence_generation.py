# -*- coding: utf-8 -*-

#==========导入需要的库==========#
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import csv
import os

 

#==========加载目录和文档==========#
'''
首先，设置主目录和一些有关文本特征的变量。
将最大序列长度设置为25，将词汇表中的最大单词数设置为12000，使用300维embeddings。
最后，从csv加载文本。文本文件是Quora Kaggle挑战的训练文件，包含大约808000个句子。
'''
BASE_DIR = './data/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'#80+万条问句
GLOVE_EMBEDDING = BASE_DIR + 'glove.6B.300d.txt'#单词->300维embedding
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 25 #最大序列长度25
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300 #embedding维度300

texts = [] #通过列表来存储句子
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader) #忽略标题行
    for values in reader: #对每一行[3][4]是训练用的句子
        if len(values[3].split()) <= MAX_SEQUENCE_LENGTH:
            texts.append(values[3])
        if len(values[4].split()) <= MAX_SEQUENCE_LENGTH:
            texts.append(values[4])
print('Found %s texts in train.csv' % len(texts)) #训练用句子个数
n_sents = len(texts)



#==========文本预处理==========#
'''
使用Keras的tokenizer和text_to_sequences函数预处理文本
'''
tokenizer = Tokenizer(MAX_NB_WORDS+1, oov_token='unk') #Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer.fit_on_texts(texts)
print('Found %s unique tokens' % len(tokenizer.word_index))

# **关键步骤** 若不能正常工作，丢弃OOV_Token
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS} # <= 从1开始
tokenizer.word_index[tokenizer.oov_token] = MAX_NB_WORDS + 1
word_index = tokenizer.word_index #word到index的字典
index2word = {v: k for k, v in word_index.items()} #index到word的字典

sequences = tokenizer.texts_to_sequences(texts)#序列的列表，列表中每个序列对应于一段输入文本
data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #序列全部填充到25维，尾补0
print('Shape of data tensor:', data_1.shape)
NB_WORDS = (min(tokenizer.num_words, len(word_index))+1) #+1 for zero padding 
print('NB_WORDS:', NB_WORDS)

data_val = data_1[775000:783000]
data_train = data_1[:775000]



#==========Word embeddings==========#
'''
使用预训练的Glove word embeddings。
创建一个矩阵，在词汇表中为每个单词对应一个embedding，
然后将这个矩阵作为权重传递给我们模型的embedding layer 
'''
embeddings_index = {}
f = open(GLOVE_EMBEDDING, encoding='utf8')

#取出word及其对应的embeddings，存入字典embeddings_index
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

glove_embedding_matrix = np.zeros((NB_WORDS, EMBEDDING_DIM)) #申请0数组，
for word, i in word_index.items():
    if i < NB_WORDS+1: #+1 for 'unk' oov token
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            glove_embedding_matrix[i] = embedding_vector
        else:
            # 在embeddings索引中找不到的单词，将是unk的embeddings
            glove_embedding_matrix[i] = embeddings_index.get('unk')
print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))



#==========VAE 模型==========#
'''
模型基于seq2seq架构，包含双向LSTM编码器和LSTM解码器。
通过 RepeatVector（max_len）函数，将每个时间步的潜在表示作为输入提供给解码器decoder。
为了避免标签的独热码表示，我们使用tf.contrib.seq2seq.sequence_loss函数，
它只需要单词索引作为标签（与embedding矩阵的输入相同）并在内部计算最终的softmax
（所以 模型以具有线性激活的dense层结束）。 
另，“sequence_loss”允许使用采样的softmax，这有助于处理大型词汇表（例如，具有50k字词汇），
但在此没有使用。这里使用的解码器与文中实现的解码器不同; 
不是将context vector作为解码器的初始状态和预测的单词作为输入，而是在每个时间步处输入潜在表示z作为输入。
'''
batch_size = 100
max_len = MAX_SEQUENCE_LENGTH
emb_dim = EMBEDDING_DIM
latent_dim = 64
intermediate_dim = 256
epsilon_std = 1.0
kl_weight = 0.01
num_sampled=500
act = ELU()


x = Input(shape=(max_len,)) #输入是按批量的25维向量(句子)
x_embed = Embedding(NB_WORDS, emb_dim, weights=[glove_embedding_matrix], input_length=max_len, trainable=False)(x)
h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
#h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
#h = Dropout(0.2)(h)
#h = Dense(intermediate_dim, activation='linear')(h)
#h = act(h)
#h = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 注意"output_shape" 并非必须对于TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# 分别实例化这些层，以便以后重用
repeated_context = RepeatVector(max_len)
decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = Dense(NB_WORDS, activation='linear')#softmax is applied in the seq2seqloss by tf #TimeDistributed()
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

#Sampled softmax
#logits = tf.constant(np.random.randn(batch_size, max_len, NB_WORDS), tf.float32)
#targets = tf.constant(np.random.randint(NB_WORDS, size=(batch_size, max_len)), tf.int32)
#proj_w = tf.constant(np.random.randn(NB_WORDS, NB_WORDS), tf.float32)
#proj_b = tf.constant(np.zeros(NB_WORDS), tf.float32)
#
#def _sampled_loss(labels, logits):
#    labels = tf.cast(labels, tf.int64)
#    labels = tf.reshape(labels, [-1, 1])
#    logits = tf.cast(logits, tf.float32)
#    return tf.cast(
#                    tf.nn.sampled_softmax_loss(
#                        proj_w,
#                        proj_b,
#                        labels,
#                        logits,
#                        num_sampled=num_sampled,
#                        num_classes=NB_WORDS),
#                    tf.float32)
#softmax_loss_f = _sampled_loss

# 用于计算VAE损失的自定义层
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

    def vae_loss(self, x, x_decoded_mean):
        #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)#,
                                                     #softmax_loss_function=softmax_loss_f), axis=-1)#,
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        xent_loss = K.mean(xent_loss)
        kl_loss = K.mean(kl_loss)
        return K.mean(xent_loss + kl_weight * kl_loss)
    
    #编写一个call方法，来实现自定义层
    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        print(x.shape, x_decoded_mean.shape)
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)
    
def kl_loss(x, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
opt = Adam(lr=0.01) 
vae.compile(optimizer='adam', loss=[zero_loss], metrics=[kl_loss])
vae.summary()



#==========模型训练===========#
'''
通过keras.fit()训练100epochs。对于验证数据，传递相同的数组两次，因为此模型的输入和标签相同。
如果不使用“tf.contrib.seq2seq.sequence_loss”（或其他类似的函数），
将必须作为标签传递单词的one-hot码高维度序列(batch_size，seq_len，vocab_size)消耗大量内存。
'''
def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5" 
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    return checkpointer

checkpointer = create_model_checkpoint('models', 'vae_seq2seq_test_very_high_std')



vae.fit(data_train, data_train,
     shuffle=True,
     epochs=100,
     batch_size=batch_size,
     validation_data=(data_val, data_val), callbacks=[checkpointer])

#print(K.eval(vae.optimizer.lr))
#K.set_value(vae.optimizer.lr, 0.01)

vae.save('models/vae_lstm.h5')
#vae.load_weights('models/vae_seq2seq_test.h5')



#==========来自潜在空间的项目和样本句子==========#
'''
构建一个编码器模型，将句子encode到潜在空间，一个解码器模型从潜在空间返回到文本表示
'''
#编码器模型
encoder = Model(x, z_mean)
#encoder.save('models/encoder32dim512hid30kvocab_loss29_val34.h5')

#一个解码器模型从潜在空间返回到文本表示
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation('softmax')(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean)



#==========在validation句子上测试==========#
index2word = {v: k for k, v in word_index.items()}
index2word[0] = 'pad'

#在validation句子上测试
sent_idx = 100
sent_encoded = encoder.predict(data_val[sent_idx:sent_idx+2,:])
x_test_reconstructed = generator.predict(sent_encoded, batch_size = 1)
reconstructed_indexes = np.apply_along_axis(np.argmax, 1, x_test_reconstructed[0])
#np.apply_along_axis(np.max, 1, x_test_reconstructed[0])
#np.max(np.apply_along_axis(np.max, 1, x_test_reconstructed[0]))
word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
print(' '.join(word_list))
original_sent = list(np.vectorize(index2word.get)(data_val[sent_idx]))
print(' '.join(original_sent))



#==========句子处理和插值==========#
#解析句子函数
def sent_parse(sentence, mat_shape):
    sequence = tokenizer.texts_to_sequences(sentence)
    padded_sent = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sent#[padded_sent, sent_one_hot]

# input: encode后的句子向量
# output: 具有最高余弦相似性的数据集中的编码句子向量
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec

# input: 两个点，整数n
# output:n个输入点之间的线上的等距点（包括）
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample

# input:原始维度句子向量
# output: 句子文本
def print_latent_sentence(sent_vect):
    sent_vect = np.reshape(sent_vect,[1,latent_dim])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_len,NB_WORDS])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w not in ['pad']]
    print(' '.join(w_list))
    #print(word_list)
     
def new_sents_interp(sent1, sent2, n):
    tok_sent1 = sent_parse(sent1, [MAX_SEQUENCE_LENGTH + 2])
    tok_sent2 = sent_parse(sent2, [MAX_SEQUENCE_LENGTH + 2])
    enc_sent1 = encoder.predict(tok_sent1, batch_size = 16)
    enc_sent2 = encoder.predict(tok_sent2, batch_size = 16)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    for point in test_hom:
        print_latent_sentence(point)



#===========示例==========#
'''
可以解析两个句子并在它们之间插入生成新句子
'''
sentence1=['gogogo where can i find a bad restaurant endend']
mysent = sent_parse(sentence1, [MAX_SEQUENCE_LENGTH + 2])
mysent_encoded = encoder.predict(mysent, batch_size = 16)
print_latent_sentence(mysent_encoded)
print_latent_sentence(find_similar_encoding(mysent_encoded))

sentence2=['gogogo where can i find an extremely good restaurant endend']
mysent2 = sent_parse(sentence2, [MAX_SEQUENCE_LENGTH + 2])
mysent_encoded2 = encoder.predict(mysent2, batch_size = 16)
print_latent_sentence(mysent_encoded2)
print_latent_sentence(find_similar_encoding(mysent_encoded2))
print('-----------------')

new_sents_interp(sentence1, sentence2, 5)



#==========结尾==========#
'''
另，
结果还不完全令人满意，有很多句子有语法问题，并且在插值中多次生成相同的句子。
后期改进：
 - 参数调整（可尝试更大的网络）
 - 更一般化的数据集（Quora句子都是问句）
'''