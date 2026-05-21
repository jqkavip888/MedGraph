
# 设定正向和反向2个标签字典
l2i_dic = {'O':0,u'B-sym':1,u'B-dis':2,u'I-sym':3,u'I-dis':4,u'E-sym':5,u'E-dis':6,'<pad>':7,'<start>':8,'<eos>':9}
i2l_dic = {0:'O',1:u'B-sym',2:u'B-dis',3:u'I-sym',4:u'I-dis',5:u'E-sym',6:u'E-dis',7:'<pad>',8:'<start>',9:'<eos>'}

# 训练集测试集词表路径
train_file = 'data/train.txt'
test_file = 'data/test.txt'
vocab_file = 'data/vocab.txt'

# 模型保存读取路径
save_model_file = 'data/model/idcnn.pt'
model_path = 'data/model/idcnn.pt'

# 一些关键超参数
max_length = 256
batch_size = 32
epochs = 80
tagset_size = len(l2i_dic)
dropout = 0.4
use_cuda = True

