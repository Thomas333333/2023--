from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
from django.http import JsonResponse

from keras.models import load_model
from keras_bert import Tokenizer
from keras_bert import get_custom_objects
import numpy as np
import codecs

from keras.layers import *
from keras.models import Model
from keras import backend as K


maxlen = 200
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
 #       dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


train_model = load_model("ChineseQAmodel.h5", custom_objects=get_custom_objects())

# 对单句话进行预测
def predict_single_text(text1,text2):
    # 利用BERT进行tokenize
    pred_ans=""
    squad_examples = []
    context = text1
    question = text2
    answer_text = "answer"
    all_answers = ["answers1", "answers1"]
    start_char_idx = 1
    squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
    squad_eg.preprocess()
    if (squad_eg.skip == False):
        squad_examples.append(squad_eg)

    x_test, y_test = create_inputs_targets(squad_examples)
    pred_start, pred_end = train_model.predict(x_test)
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        squad_eg = squad_examples[idx]
        offsets = squad_eg.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
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




def outputStr(str):
    text1 = "在Transformer中，多头自注意力用于表示输入序列和输出序列，不过解码器必须通过掩蔽机制来保留自回归属性。Transformer的优点是并行性非常好，符合目前的硬件（主要指GPU）环境。"
    text2 = "Transformer有什么优点？"
    y = predict_single_text(text1, text2)
    print(y)
    rstr="返回内容是："+y
    return rstr


# Create your views here.

@csrf_exempt
def getanswer(request):
 print("start getanswer:")
 post_content=json.loads(request.body, encoding='utf-8')['content']
# post_content = json.loads(request.body, encoding='utf-8')['content']
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