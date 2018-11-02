# definition-modeling

1. 首先将GoogleNews-vectors-negative300.bin文件放在data/word2vec文件夹下，运行utils文件夹下preprocess.py进行数据预处理。运行成功会在data/processed文件夹下生成处理好的文件。
2. CUDA_VISIBLE_DEVICES=  python train.py -–cuda进行训练。模型参数可在train.py里进行修改。每次会保存最好的模型在checkpoints文件夹下。
3. Generate.py在gen文件夹下生成测试集的定义。
4. Eval文件夹下的bleu.py进行BLEU的测试。

### 之前数据格式：
\<s>measuring instrument designed to measure power\<s/>
#### 被定义词dynamometer的embedding作为rnn的 intial hidden state。
### 现在数据格式：
dynamometer\<def>measuring instrument designed to measure power\<s/>
#### 将被定义词放在定义序列开头，rnn的 intial hidden state置0。
### 之前的结果：
　|Paper(PPL)）|Me(PPL)|Paper(BLEU)|Me(BLEU)
---|:--:|---:|-:|-
Seed|56.350|58.041|30.46| 37.03
S+I|57.372|60.762|31.58| 37.76
S+H|58.147|62.517|29.66| **38.27**
S+G|50.949|57.414|34.72| 37.03
S+G+CH|48.566|**56.512**|**35.78**| 37.66
S+G+CH+HE|**48.168**|56.762|35.28| 34.68
### 现在的结果：
　|Paper(PPL)）|Me(PPL)|Paper(BLEU)|Me(BLEU)
---|:--:|---:|-:|-
Seed|56.350|34.55|30.46| 34.20
S+I|57.372|35.64|31.58| 34.59
S+H|58.147|38.32|29.66| 35.82
S+G|**50.949**|**34.16**|**34.72**| **37.84**
S+G+CH|n|n|n| n
S+G+CH+HE|n|n|n| n
