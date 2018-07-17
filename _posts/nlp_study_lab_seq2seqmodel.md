---
layout: post

title: "nlp_seq2seq_tutorial"

author: "hyeju.kim"

categories: NLP

tags: [NLP]

image: 
---



# Seq2Seq Tutorial : 일정한 날짜 형식으로 반환하기 

본 포스트는 [여기](https://github.com/sachinruk/deepschool.io/blob/master/DL-Keras_Tensorflow/Lesson%2019%20-%20Seq2Seq%20-%20Date%20translator.ipynb) 내용을 바탕으로 작성한 것입니다.

자세히 코드로 들어가기 전에 seq2seq 모델을 살펴보자

![seq2seq](https://user-images.githubusercontent.com/32008883/42751807-f18baf98-8926-11e8-97c5-5f1ea327e572.png)

seq2seq 모델은 두개의 lstm 모델(encoder,decoder부분)을 가지고 있다. encoder는 many to one lstm 으로 인풋값들이 하나의 hidden value을 만들게 된다. decoder쪽은 many to many 구조인데, 바로 앞에서 나온 결과 값이 그다음에도 반영이 되므로 many to many 구조이다. 


```python
! pip install tensorflow
```

    Collecting tensorflow
      Using cached https://files.pythonhosted.org/packages/e7/88/417f18ca7eed5ba9bebd51650d04a4af929f96c10a10fbb3302196f8d098/tensorflow-1.9.0-cp36-cp36m-win_amd64.whl
    Collecting setuptools<=39.1.0 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/8c/10/79282747f9169f21c053c562a0baa21815a8c7879be97abd930dbcf862e8/setuptools-39.1.0-py2.py3-none-any.whl
    Collecting grpcio>=1.8.6 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/d5/c6/15728549704f9c03db7179b7f99303b91b7703e18a50f5e7b47e59b289ea/grpcio-1.13.0-cp36-cp36m-win_amd64.whl
    Requirement already satisfied: absl-py>=0.1.6 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (0.2.2)
    Collecting protobuf>=3.4.0 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/75/7a/0dba607e50b97f6a89fa3f96e23bf56922fa59d748238b30507bfe361bbc/protobuf-3.6.0-cp36-cp36m-win_amd64.whl
    Requirement already satisfied: six>=1.10.0 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (1.11.0)
    Collecting gast>=0.2.0 (from tensorflow)
    Requirement already satisfied: numpy>=1.13.3 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (1.14.5)
    Collecting tensorboard<1.10.0,>=1.9.0 (from tensorflow)
      Using cached https://files.pythonhosted.org/packages/9e/1f/3da43860db614e294a034e42d4be5c8f7f0d2c75dc1c428c541116d8cdab/tensorboard-1.9.0-py3-none-any.whl
    Requirement already satisfied: wheel>=0.26 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (0.31.1)
    Collecting termcolor>=1.1.0 (from tensorflow)
    Requirement already satisfied: astor>=0.6.0 in c:\programdata\anaconda3\lib\site-packages (from tensorflow) (0.7.1)
    Requirement already satisfied: werkzeug>=0.11.10 in c:\programdata\anaconda3\lib\site-packages (from tensorboard<1.10.0,>=1.9.0->tensorflow) (0.14.1)
    Requirement already satisfied: markdown>=2.6.8 in c:\programdata\anaconda3\lib\site-packages (from tensorboard<1.10.0,>=1.9.0->tensorflow) (2.6.11)
    Installing collected packages: setuptools, grpcio, protobuf, gast, tensorboard, termcolor, tensorflow
      Found existing installation: setuptools 39.2.0
        Uninstalling setuptools-39.2.0:


    distributed 1.22.0 requires msgpack, which is not installed.
    Could not install packages due to an EnvironmentError: [WinError 5] 액세스가 거부되었습니다: 'c:\\programdata\\anaconda3\\lib\\site-packages\\__pycache__\\easy_install.cpython-36.pyc'
    Consider using the `--user` option or check the permissions.


​    


```python
import tensorflow as tf
```


    ---------------------------------------------------------------------------
    
    ModuleNotFoundError                       Traceback (most recent call last)
    
    <ipython-input-1-64156d691fe5> in <module>()
    ----> 1 import tensorflow as tf


    ModuleNotFoundError: No module named 'tensorflow'



```python
! pip install faker babel
```

    Collecting faker
      Downloading https://files.pythonhosted.org/packages/13/c7/e032cdedd0a3814ce2978aa543f7e30c6b32434f9f84b495cd574027a207/Faker-0.8.17-py2.py3-none-any.whl (753kB)
    Requirement already satisfied: babel in c:\programdata\anaconda3\lib\site-packages (2.5.3)
    Requirement already satisfied: python-dateutil>=2.4 in c:\programdata\anaconda3\lib\site-packages (from faker) (2.7.3)
    Collecting text-unidecode==1.2 (from faker)
      Downloading https://files.pythonhosted.org/packages/79/42/d717cc2b4520fb09e45b344b1b0b4e81aa672001dd128c180fabc655c341/text_unidecode-1.2-py2.py3-none-any.whl (77kB)
    Requirement already satisfied: six>=1.10 in c:\programdata\anaconda3\lib\site-packages (from faker) (1.11.0)
    Requirement already satisfied: pytz>=0a in c:\programdata\anaconda3\lib\site-packages (from babel) (2018.4)
    Installing collected packages: text-unidecode, faker
    Successfully installed faker-0.8.17 text-unidecode-1.2


    distributed 1.21.8 requires msgpack, which is not installed.



```python
! pip install tqdm
```

    Collecting tqdm
      Downloading https://files.pythonhosted.org/packages/93/24/6ab1df969db228aed36a648a8959d1027099ce45fad67532b9673d533318/tqdm-4.23.4-py2.py3-none-any.whl (42kB)
    Installing collected packages: tqdm
    Successfully installed tqdm-4.23.4


    distributed 1.21.8 requires msgpack, which is not installed.



```python
! pip install utilities
```

    Collecting utilities


      Could not find a version that satisfies the requirement utilities (from versions: )
    No matching distribution found for utilities



```python
! pip install utilities
```

    Collecting utilities


      Could not find a version that satisfies the requirement utilities (from versions: )
    No matching distribution found for utilities



```python
! pip install scipy
```


```python
#필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import random
import json
import os
import time

from faker import Faker
import babel
from babel.dates import format_date

import tensorflow as tf

import tensorflow.contrib.legacy_seq2seq as seq2seq
#from utilities import show_graph

from sklearn.model_selection import train_test_split
```


    ---------------------------------------------------------------------------
    
    ImportError                               Traceback (most recent call last)
    
    <ipython-input-4-ba7bc740c3d6> in <module>()
         18 #from utilities import show_graph
         19 
    ---> 20 from sklearn.model_selection import train_test_split


    C:\ProgramData\Anaconda3\envs\tensorflow\lib\site-packages\sklearn\__init__.py in <module>()
        132 else:
        133     from . import __check_build
    --> 134     from .base import clone
        135     __check_build  # avoid flakes unused variable error
        136 


    C:\ProgramData\Anaconda3\envs\tensorflow\lib\site-packages\sklearn\base.py in <module>()
          9 
         10 import numpy as np
    ---> 11 from scipy import sparse
         12 from .externals import six
         13 from .utils.fixes import signature


    ImportError: No module named 'scipy'


### 데이터 셋 만들기

다음은 튜토리얼에 사용할 fake 데이터 셋을 만드는 과정이다. 지금은 생략해도 될 듯하다.


```python
fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

# change this if you want it to work with only a single language
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]
```


```python
def create_date():
    """
        Creates some fake dates 
        :returns: tuple containing 
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0,3) # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine #, dt

data = [create_date() for _ in range(50000)]
```

생성된 데이터를 살펴보면, 왼쪽 부분들은 input으로 들어갈 날짜 데이터이다. 이렇게 다양한 날짜 데이터를 오른쪽 날짜 형태로 바꾸는 것이 목표이다. 


```python
data[:10]
```




    [('7 07 13', '2013-07-07'),
     ('30 JULY 1977', '1977-07-30'),
     ('Tuesday, 14 September 1971', '1971-09-14'),
     ('18 09 88', '1988-09-18'),
     ('31, Aug 1986', '1986-08-31'),
     ('10/03/1985', '1985-03-10'),
     ('Sunday, 1 July 1979', '1979-07-01'),
     ('22 December, 1976', '1976-12-22'),
     ('tuesday, january 19, 2016', '2016-01-19'),
     ('FEBRUARY 11 2007', '2007-02-11')]



### 주어진 데이터를 숫자로 바꾸는 코드


```python
x = [x for x, y in data]
y = [y for x, y in data]

u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))
```


```python
u_characters
```




    {' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}




```python
char2numY
```




    {'3': 0,
     '8': 1,
     '5': 2,
     '-': 3,
     '1': 4,
     '7': 5,
     '0': 6,
     '4': 7,
     '6': 8,
     ' ': 9,
     '9': 10,
     '2': 11}



### 시퀀스 길이 맞추기 - 빈자리에 <P\AD> 넣기 


```python

char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']]*(max_len - len(date)) +[char2numX[x_] for x_ in date] for date in x]
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)
```

    <PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD>31, Aug 1986



```python
print(''.join([num2charX[x_] for x_ in x[2]]))
x = np.array(x)
```

    <PAD><PAD><PAD>Tuesday, 14 September 1971


### DECODER에 들어갈 데이터 앞에 <G\O> 붙이기


```python
char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)
```

    <GO>1986-08-31



```python
x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1
print(x_seq_length, y_seq_length)
```

    29 10



```python
def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size
```


```python
epochs = 2
batch_size = 128
nodes = 32
embed_size = 10

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)

#학습시킬 때 encoding의 last_state는 decoding의 initial state가 된다.  
with tf.variable_scope("decoding") as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)
#connect outputs to 
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(char2numY), activation_fn=None) 
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
```


```python
dec_outputs.get_shape().as_list()
```


```python
last_statelast_sta [0].get_shape().as_list()
```


```python
inputs.get_shape().as_list()
```


```python
date_input_embed.get_shape().as_list()
```


```python
show_graph(tf.get_default_graph().as_graph_def())
```

### training 시키기


```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
```


```python
sess.run(tf.global_variables_initializer())
epochs = 10
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
            feed_dict = {inputs: source_batch,
             outputs: target_batch[:, :-1],
             targets: target_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, 
                                                                      accuracy, time.time() - start_time))
```

### test 해보기


```python
source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                feed_dict = {inputs: source_batch,
                 outputs: dec_input})
    prediction = batch_logits[:,-1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:,None]])
    
print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))
```


```python
num_preds = 2
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))
```


```python
source_batch[0]
```
