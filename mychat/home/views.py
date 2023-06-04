from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from predict_with_generate import T5PegasusTokenizer

from tensorflow import keras
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from keras import backend as K
from keras.layers import *
from keras.optimizers import Adam
from elasticsearch import Elasticsearch
import json
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import jieba
import tensorflow as tf

vocab_size=2100 # sql编程
sequence_length=300 # sql编程

algOpTextEn=['=','<>','>=','<=','>','<','like']
algOpTextZh=['等于','不等于','大于等于','小于等于','大于','小于','类似于']
cateOpEn=['=']
cateOpZh=['为']

sample={}

chat_history = []

def getAnswer(request):

    answer = None
    if request.method == 'POST':
        post_content = request.POST['content']
        answer = outputStr(post_content)
        chat_history.append((post_content, answer))  # append question and answer to chat_history
    return render(request, 'runoob.html', {'answer': answer, 'chat_history': chat_history})

def clearHistory(request):
    chat_history.clear()
    return HttpResponse()

def runoob(request):
    results = []  # 初始结果为空
    answer = ""  # 初始回答为空

    if request.method == "POST":
        content = request.POST['content']  # 获取用户输入的问题
        answer = outputStr(content)  # 调用 getAnswer 函数并将回答存储在 answer 变量中

    context = {
        'results': results,
        'answer': answer,

    }

    return render(request, 'runoob.html', context)


#SQL programming

def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return [-1]

def textReplace(text, s):
    newtext=text
    if (s>0):  #requirement preprocessing sql
        posv = find_all(sample['tablezh'], newtext)
        pos1= posv[-1]
        if (pos1>=0):
            newtext=newtext[:pos1]+newtext[pos1:].replace(sample['tablezh'],'table')
        latt=len(sample['attribute'])
        for i in range(latt):
            pos1 = newtext.find(sample['attribute'][i]['zh'])
            if (pos1 >= 0):
                newtext = newtext[:pos1]+newtext[pos1:].replace(sample['attribute'][i]['zh'], 'att'+str(i))
        #calculate OP processing
        algOpNum=len(algOpTextZh)
        for i in range(algOpNum):
            posv=find_all(algOpTextZh[i], newtext)
            if (len(posv)==1 and posv[0]<0): continue
            else:
                offset=0
                li=len(algOpTextZh[i])
                for j in range(len(posv)):
                    pos1=posv[j]+offset
                    if (pos1>0 and newtext[pos1-1:pos1].isdigit() and newtext[pos1+li:pos1+li+1].isdigit()):
                        pos1+=li
                        for j in range(5):
                            if (newtext[pos1+j].isdigit()): continue
                            else: break
                        digitValue=newtext[pos1:pos1+j]
                        pos2=newtext[:pos1+j].rfind("att")
                        if (pos2>=0):
                            index=newtext[pos2+3:pos1-len(algOpTextZh[i])]
                            newtext=newtext[:pos1]+newtext[pos1:pos1+j+1].replace(digitValue, "value"+index)+newtext[pos1+j+1:]
                            offset+=len("value"+index)-len(digitValue)
                            sample['attribute'][int(index)]['value']=str(digitValue)

        #category OP processing
        cateOpNum=len(cateOpZh)
        for i in range(cateOpNum):
            pos1=newtext.find(cateOpZh[i])
            cateValue=""
            if (pos1>=0):
                newtext1=newtext[:pos1]
                newtext2=newtext[pos1:]
                # pos1+=len(cateOpZh[i])
                pos2=newtext2.find("'")
                for j in range(1,5):
                    if (newtext2[pos2+1+j]=="'"): break
                cateValue=newtext2[pos2+1:pos2+1+j]
                pos4 = newtext1.rfind("att")
                if (pos4>=0):
                    index=newtext1[pos4+3:]
                    newtext=newtext1+newtext2[pos2-len(cateOpZh[i]):].replace(cateValue, "value"+index)
                    sample['attribute'][int(index)]['value']="'"+cateValue+"'"

    else:   #sql statement preprocessing
        pos1=newtext.find(sample['tableen'])
        if (pos1>=0):
            newtext=newtext.replace(sample['tableen'],'table')
        latt=len(sample['attribute'])
        for i in range(latt):
            pos1 = newtext.find(sample['attribute'][i]['en'])
            if (pos1 >= 0):
                newtext = newtext.replace(sample['attribute'][i]['en'], 'att'+str(i))

        #calculate OP processing
        algOpNum=len(algOpTextEn)
        for i in range(algOpNum):
            posv=find_all(algOpTextEn[i], newtext)
            if (len(posv)==1 and posv[0]<0): continue
            else:
                li = len(algOpTextEn[i])
                for j in range(len(posv)):
                    pos1=posv[j]
                    if (pos1>0 and newtext[pos1-1:pos1].isdigit() and (newtext[pos1+li:pos1+li+1].isdigit() or newtext[pos1+li:pos1+li+1]=="'")):
                        pos1+=li
                        pos2=newtext[:pos1].rfind("att")
                        if (pos2>=0):
                            index=newtext[pos2+3:pos1-len(algOpTextEn[i])]
                            if (sample['attribute'][int(index)]['type']=='int'):
                                digitValue= str(sample['attribute'][int(index)]['value'])
                                newtext=newtext[:pos1]+newtext[pos1:pos1+len(digitValue)+1].replace(digitValue, "value"+index)+newtext[pos1+len(digitValue)+1:]
                            else:
                                cateValue = sample['attribute'][int(index)]['value']
                                newtext = newtext[:pos1]+newtext[pos1:pos1+len(cateValue)+1].replace(cateValue,"'value" + index + "'")+newtext[pos1+len(cateValue)+1:]

    return newtext




def standarizeRequirement(text,sqltext):
    newtext="在表格"
    newsql="select"

    textlist = text.split(',')
    text = textlist[0].strip()
    pos1 = text.find('表格')
    pos2 = text.find('(')
    pos3 = text.find(')')
    sample['tablezh'] = text[pos1+2:pos2]
    sample['tableen'] = text[pos2 + 1: pos3]
    newtexttable='table'+text[pos3+1:]+","
    attnum = len(textlist)
    att = []
    newtextatt=""
    for i in range(1,attnum - 1):   #previous attnum -2
        attelem = {}
        text = textlist[i].strip()
        if(i==1):
            pos1=text.find('属性有')
            text=text[pos1+3:]
            #newtextatt+= text[:pos1+3]
        pos2=0
        for j in range(len(text)):
            if text[j].isascii()==True:
                break
            else:
                pos2+=1
        attelem['zh']=text[:pos2]
        temptext = text[pos2:]
        pos3 = temptext.find('(')
        attelem['en']=temptext[:pos3]
        attelem['type'] = temptext[pos3 + 1:-1]
        attelem['order'] = i
        att.append(attelem)
        newtextatt+="att"+str(i-1)+","
    sample['attribute'] = att

    text = textlist[attnum-1].strip()
    if text.find('sql')>0:
        newlasttext =textReplace(text, 1)
        newtext=newtext+newtexttable+newtextatt+newlasttext
        sqltext =textReplace(sqltext,0)
    else:
        newtext='输入文本不符合规范'
    return newtext, sqltext

def custom_standardization(input_string):  # 自定义标准化函数
    # input_string = input_string.replace('[','')
    # input_string = input_string.replace(']', '')
    lowercase=tf.strings.lower(input_string)  # 先转成小写
    # return lowercase
    return tf.strings.regex_replace(lowercase,'[\[\]]',"")  # 去掉[和])
    # return tf.strings.regex_replace(lowercase,f'[{re.escape(strip_chars)}]')  # 保留[和]，去掉¿


source_vectorization=TextVectorization(max_tokens=vocab_size,output_mode='int',standardize=custom_standardization, output_sequence_length=sequence_length)
# 源语言（英语）的词嵌入

target_vectorization=TextVectorization(max_tokens=vocab_size,output_mode='int',standardize=custom_standardization, output_sequence_length=sequence_length+1)
# 目标语言（中文）的词嵌入，生成的中文语句子多了一个词元，因为在训练的时候需要将句子偏移一个时间步

source_vocab_file= "source_vocab.json"
target_vocab_file= "target_vocab.json"

json_file = open(source_vocab_file, 'r', encoding='utf-8')
source_vocab = json.load(json_file)
json_file.close()

source_vectorization.set_vocabulary(source_vocab)

json_file = open(target_vocab_file, 'r', encoding='utf-8')
target_vocab = json.load(json_file)
json_file.close()
target_vectorization.set_vocabulary(target_vocab)


batch_size=32


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config=super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads":self.num_heads,
            "dense_dim":self.dense_dim,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    def get_config(self):
        config=super().get_config()
        config.update({
            "embed_dim":self.embed_dim,
            "sequence_length": self.sequence_length,
            "vocab_size":self.vocab_size,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config=super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "latent_dim": self.latent_dim,
        })
        return config

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

embed_dim = 256
latent_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="chinese")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
#encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="sql")
#encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
#decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

#decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs)

transformer.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

transformer.load_weights('TransformerSQLModelWeights2.h5')

import numpy as np

target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))
max_decoded_sentence_length = 60  #20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        #predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        next_token_predictions = transformer.predict([tokenized_input_sentence, tokenized_target_sentence])

        #sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        # if sampled_token == "[end]":
        if sampled_token.find("endend")>=0:
            break
    return decoded_sentence


def deStandarize(text):
    tablename=sample['tableen']
    text1 = text.replace("from table ", "from "+tablename+" ")
    text=text1
    try:
        posv=find_all("att",text)
        lenatt=len("att")
        lp=len(posv)
        postfix = []
        for i in range(lp): postfix.append(text[posv[i]+lenatt:posv[i] + lenatt+1])
        postfix = set(postfix)
        postfix=list(postfix)
        offset=0
        for i in range(lp):
            pos1 = posv[i]+offset
            if(pos1<0): break
            for j in range(1,5):
                if (pos1 >=0 and text[pos1+lenatt+j:pos1+lenatt+j].isdigit()): continue
                else: break
            attindex=text[pos1+lenatt:pos1+lenatt+j].strip()
            if (postfix.count(attindex)<=0):continue
            atttemp="att"+attindex
            attname=sample['attribute'][int(attindex)]['en']
            text = text.replace(" "+atttemp+" ", " "+attname+" ")
            offset+=len(attname)-len(atttemp)

        posv=find_all("value",text)
        lenvalue=len("value")
        lp=len(posv)
        offset=0
        for i in range(lp):
            pos1 = posv[i]+offset
            if(pos1<0): break
            for j in range(1,5):
                if (pos1 >=0 and text[pos1+lenvalue+j:pos1+lenvalue+j].isdigit()): continue
                else: break
            valueindex=text[pos1+lenvalue:pos1+lenvalue+j].strip()
            valuetemp="value"+valueindex
            if ('value' in sample['attribute'][int(valueindex)].keys()):
                valuereal=sample['attribute'][int(valueindex)]['value']
            else:
                valuereal="wrongindex"
            text = text.replace(valuetemp, valuereal)
            offset+=len(valuereal)-len(valuetemp)
    except:
        text=text1
    return text


def testtext(text):
    temptext=text
    fulltext,sqltext =standarizeRequirement(temptext,"")
    print(fulltext)
    splits = jieba.cut(fulltext.strip(), cut_all=False)
    # splits = [term.encode("utf8", "ignore") for term in splits]
    text = ""
    for split in splits:
        text += split + " "
    chinese = text.rstrip()
    translated = decode_sequence(chinese)
    print(translated)
    # translated =translated.replace("startstart","")
    translated = translated.replace("[start]", "")
    # translated = translated.replace("select all ", "select * ")
    pos=translated.find(" endend")
    if (pos>=0):
        translatedtemp=translated[:pos]
    else:
        translatedtemp="wrong output"

    decode_text=deStandarize(translatedtemp)
    decode_text = decode_text.replace("select all ", "select * ")
    return decode_text



# ES access
host="http://localhost:9200"
es = Elasticsearch(host, maxsize=15)

# BERT model
maxlen = 200

config_path = 'bert_config.json'
checkpoint_path = 'bert_model.ckpt'
dict_path = 'vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.input_ids=[]
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # if ("五" in answer_text):
        #     i=1

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)[0]
        self.context_token_to_char = tokenized_context
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for i in range(start_char_idx,end_char_idx):
            ans_token_idx.append(i)
        # for idx in tokenized_context:
        #     if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
        #         ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        self.start_token_idx = ans_token_idx[0]
        self.end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)[0]


        # Create inputs
        self.input_ids = tokenized_context + tokenized_question[1:]
        self.token_type_ids = [0] * len(tokenized_context) + [1] * len(
            tokenized_question[1:]
        )
        self.attention_mask = [1] * len(self.input_ids)
        if (len(self.input_ids)>maxlen):
            dist = 20
            a = maxlen - len(tokenized_question[1:])
            if (a > self.start_token_idx + dist and a > self.end_token_idx + dist):
                self.input_ids = tokenized_context[:a - 1] + tokenized_question[1:]
                self.token_type_ids = [0] * len(tokenized_context[0:a - 1]) + [1] * len(
                    tokenized_question[1:]
                )
                self.attention_mask = [1] * len(self.input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed

        padding_length = maxlen - len(self.input_ids)
        if padding_length > 0:  # pad
            self.input_ids =self.input_ids + [0] * padding_length
            self.attention_mask = self.attention_mask + [0] * padding_length
            self.token_type_ids = self.token_type_ids + [0] * padding_length
            return
        elif padding_length < 0:  # skip
            self.skip = True
            return


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
   l.trainable = True

##QA model
def create_QA_model():
    input_ids = Input(shape=(None,)) # 待识别句子输入
    token_type_ids = Input(shape=(None,)) # 待识别句子输入
    attention_mask = Input(shape=(None,)) # 实体左边界（标签）
    # embedding = bert_model([input_ids, token_type_ids, attention_mask])
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(input_ids)

    embedding = bert_model([input_ids, token_type_ids])

    start_logits = Dense(1, name="start_logit", use_bias=False)(embedding)
    # start_logits = Flatten()(start_logits)
    start_logits = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([start_logits, x_mask])

    end_logits = Dense(1, name="end_logit", use_bias=False)(embedding)
    #end_logits = Flatten()(end_logits)
    end_logits = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([end_logits, x_mask])

    start_probs = Activation(keras.activations.softmax)(start_logits)
    end_probs = Activation(keras.activations.softmax)(end_logits)

    train_model = keras.Model(
        inputs=[input_ids, token_type_ids],
        outputs=[start_probs, end_probs],
    )
    qaloss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    qaoptimizer = keras.optimizers.Adam(lr=5e-5)
    train_model.compile(optimizer=qaoptimizer, loss=[qaloss, qaloss])
    return train_model

keras.backend.clear_session()
QAmodel = create_QA_model()
QAmodel.load_weights('ChineseQAweights.hdf5')


bert_model2 = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model2.layers:
   l.trainable = True

#similar sentence
def create_model():
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model2([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    simp = Dense(1, activation='sigmoid')(x)

    sim_model = keras.Model([x1_in, x2_in], simp)
    sim_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5), # 用足够小的学习率
        metrics=['accuracy']
    )
    return sim_model

keras.backend.clear_session()
simSentModel = create_model()
simSentModel.load_weights('4_ChineseSimSentWeights.hdf5')

def predict_similar_text(senttext1,senttext2):
    # 利用BERT进行tokenize
    senttext1 = senttext1[:maxlen]
    senttext2 = senttext2[:maxlen]

    x1, x2 = tokenizer.encode(first=senttext1,second=senttext2)
    X1 = x1 + [0] * (maxlen - len(x1)) if len(x1) < maxlen else x1
    X2 = x2 + [0] * (maxlen - len(x2)) if len(x2) < maxlen else x2

    # 模型预测并输出预测结果
    predicted = simSentModel.predict([np.array([X1]), np.array([X2])])[0]
    y1=predicted[0]
    print(senttext2+":"+str(y1))
    return y1

'''T5'''
t5_tokenizer = T5PegasusTokenizer.from_pretrained('chinese_t5_pegasus_base')
import torch

gen_model = torch.load('saved_model/summary_model')


def generate_answer_from_text(text, tokenizer, model):
    length = len(text)
    if length < 50:
        max_gen_len = 20
    elif 50 <= length <= 100:
        max_gen_len = 50
    elif 100 < length <= 200:
        max_gen_len = 80
    else:
        max_gen_len = 100
    print(0)
    input_ids = tokenizer.encode(text, truncation='only_first')
    print(1)
    input_ids = torch.tensor(input_ids).cuda(0)
    print(2)
    input_ids = input_ids.reshape(1, -1)
    gen = model.generate(input_ids, max_length=max_gen_len)
    print(3)
    gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
    print(4)
    gen = [item.replace(' ', '') for item in gen]

    return ''.join(gen)


# 对单句话进行预测
def predict_single_text(text1,text2):
    # 利用BERT进行tokenize
    pred_ans=""
    squad_examples = []
    context = text1
    question = text2
    print(question)
    answer_text = "answer"
    all_answers = ["answers1", "answers1"]
    start_char_idx = 1
    squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
    squad_eg.preprocess()
    if (squad_eg.skip == False):
        squad_examples.append(squad_eg)

    x_test, y_test = create_inputs_targets(squad_examples)
    pred_start, pred_end = QAmodel.predict(x_test)
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        squad_eg = squad_examples[idx]
        offsets = squad_eg.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        print(start)
        print(end)
        # adjust the answer scope
        if (start > end):
            temp = end
            end = start
            start = temp - 1
        else:
            end += 1

        if start >= len(offsets):
            continue
        # pred_char_start = offsets[start][0]
        if end < len(offsets):
            #    pred_char_end = offsets[end][1]
            #     pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            pred_ans = squad_eg.context[start:end]
        else:
            #    pred_ans = squad_eg.context[pred_char_start:]
            pred_ans = squad_eg.context[start:]
    return pred_ans




def es_search_body(value, key):
    """
    将传参封装为es查询的body，可根据实际需求，判断是否新增此函数
    :param value:
    :param key:
    :return:
    """
    body = {
        "_source": ["question", "answer"],
        "query": {
            "match": {
                key: value
            }
        }
    }
    return body

# def checkSimilarQuestion(quesstr):
#     returnText="在问答库中没有找到答案"
#     key="question"
#     senttext1=quesstr   #"解释exists查询的作用"
#     bd=es_search_body(senttext1, key)
#     print(senttext1)
#     results=es.search(body=bd,index='question_answer')
#     l=len(results['hits']['hits'])
#     print(f'循环个数:{l}')
#     for i in range(l):
#         senttext2=results['hits']['hits'][i]['_source']['question']
#         y=predict_similar_text(senttext1, senttext2)
#         if(y>=0.8):
#             print(senttext2+"  score: "+str(y))
#             returnText=results['hits']['hits'][i]['_source']['answer']+"  （来自问答："+senttext2+" )"
#             print(returnText)
#             break
#     return returnText

def checkSimilarQuestion(quesstr):
    returnText="在问答库中没有找到答案"
    key="question"
    senttext1=quesstr   #"解释exists查询的作用"
    # print('问题',senttext1)
    bd=es_search_body(senttext1, key)
    print(senttext1)
    results=es.search(body=bd,index='question_answer')
    l=len(results['hits']['hits'])
    print(f'循环个数:{l}')
    # print('所有答案的数组', results)
    temp_candidate = []
    for i in range(l):
        senttext2=results['hits']['hits'][i]['_source']['question']
        y=predict_similar_text(senttext1, senttext2)

        # if(y>=0.8):
        if (y >= 0.3):
            # print(senttext2+"  score: "+str(y))
            temp_candidate.append([results['hits']['hits'][i]['_source']['answer']+"  （来自问答："+senttext2+" )",y])
            # returnText=results['hits']['hits'][i]['_source']['answer']+"  （来自问答："+senttext2+" )"
            # print(returnText)
            # break
    print('排序前的数组', temp_candidate)
    if temp_candidate:
        sorted_candidate = sorted(temp_candidate, key=lambda x: x[1],reverse=True)
        print('排序并根据阈值筛除的数组', sorted_candidate)
        returnText=sorted_candidate[0][0]
        print('最后的返回结果',returnText)

    return returnText

def checkSimilarQuestion2(quesstr):
    returnText="在问答库中没有找到答案"
    key="question"
    senttext1=quesstr   #"解释exists查询的作用"
    # print('问题',senttext1)
    bd=es_search_body(senttext1, key)
    print(senttext1)
    results=es.search(body=bd,index='question_answer')
    l=len(results['hits']['hits'])
    print(f'循环个数:{l}')
    # print('所有答案的数组', results)
    temp_candidate = []
    for i in range(l):
        senttext2=results['hits']['hits'][i]['_source']['question']
        y=predict_similar_text(senttext1, senttext2)
        if (y >= 0.3):
            temp_candidate.append([results['hits']['hits'][i]['_source']['answer'],y])
    if len(temp_candidate)==0:
        return returnText
    elif len(temp_candidate)==1:
        return temp_candidate[0][0]
    else:
        combination = ''
        for answers,_ in temp_candidate:
            combination=combination+answers
        returnText = predict_single_text(combination.lstrip(), quesstr)

    return returnText

#寻找b子串在a中的所有位置
def find_all_occurrences(a, b):
    occurrences = []
    start_index = 0
    while True:
        index = a.find(b, start_index)
        if index == -1:
            break
        occurrences.append(index)
        start_index = index + 1
    return occurrences
#将text进行去重
def deleteRepeatedWords(text):
    feature = text[-2:]
    start_index = find_all_occurrences(text, feature)
    if len(start_index)<2:
        return text
    feature_len = len(feature)
    new_feature = text[start_index[-2]+feature_len:]
    new_feature_len = len(new_feature)
    processd_text = text[:start_index[1]]
    while processd_text[-1-new_feature_len] in new_feature:
        processd_text=processd_text[:-1]
    # print(start_index)
    return deleteRepeatedWords(processd_text)


def outputStr(instr):
    text=instr.strip()
    print('输入',text)
    print('输入的类型', type(text))
    rstr = "请把问题再描述详细一点。"
    sqlpos=text.find("sql语句")
    tablepos=text.find("在表格")
    print(sqlpos)
    if(sqlpos>=0 and tablepos>=0):
        rstr = testtext(text)
        return rstr		
    pos=text.find("问答库")
    print(pos)
    if (pos>=0):
        rstr=checkSimilarQuestion(text[6:])
        return rstr
    elif(len(text)>2):

        if text.find('概括')>=0:
            text4 = text.split("概括")
            text4short = text4[:-1]
            # print(text4short[0])
            pos = text4short[0].rfind("。")
            print(text4short[0][0:pos+1])
            answer = generate_answer_from_text(text4short[0][0:pos+1], t5_tokenizer, gen_model)
            print(answer)
            return deleteRepeatedWords(answer)
        elif text.find('综合模式')>=0:
            pos = text.rfind('：')
            test5 = text[pos+2:]
            rstr = checkSimilarQuestion2(test5)
            return rstr
        text2=text.split("问")
        ques=text2[-1]
        text1=text2[:-1]
        pos=text1[0].rfind("。")
        context=text1[0][0:pos+1]
        rstr=predict_single_text(context,ques)
        print(context+" : "+ques)
    return rstr



#load models in the first time
text="在问答库中，vue.js是什么类型的框架"
rstr=checkSimilarQuestion(text)
text1 = "在Transformer中，多头自注意力用于表示输入序列和输出序列，不过解码器必须通过掩蔽机制来保留自回归属性。Transformer的优点是并行性非常好，符合目前的硬件（主要指GPU）环境。"
text2 = "Transformer有什么优点？"
rstr=predict_single_text(text1,text2)



# Create your views here.

@csrf_exempt
def getanswer(request):
 print("start getanswer:")
 # post_content=json.loads(request.body.decode('utf-8'))['content']
 post_content = json.loads(request.body.decode('utf-8'))['content']

 # post_content = json.loads(request.body)['content']

 print(type(post_content))
 post_content=outputStr(post_content)
 print(post_content)
# return JsonResponse({'content1': 'post请求'+post_content})
 return HttpResponse(post_content)

def index(request):
    #name = "Hello DTL!"
    data = {}
    data['name'] = "Tanch"
    data['message'] = "你好"
    # return render(request,"模板文件路径",context={字典格式:要在客户端中展示的数据})
    # context是个字典
    #return render(request,"./index.html",context={"name":name})
    return render(request,"./index.html",data)


def book_list(request):
	return HttpResponse("book content")