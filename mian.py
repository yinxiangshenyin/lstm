import tensorflow as tf
import deep_learning_network

import data_process
import lstm_paramter


encode_data_path='data\pinyin_target.txt'
target_data_path='data\chinese_train.txt'
encode_data_dict_path='data\pinyin_dictionary.txt'
target_data_dict_path='data\chinese_dictionary.txt'
lstm_data=data_process.ReadDate_LSTM(encode_data_path,target_data_path,encode_data_dict_path,target_data_dict_path)


paramter=lstm_paramter.lstm_paramter()
paramter.encode_input_class=lstm_data.input_data_dict
paramter.decode_input_class=lstm_data.target_data_dict


encode_input=tf.placeholder(tf.int32,shape=[paramter.batch_size,None],name="encode_input")
decode_input=tf.placeholder(tf.int32,shape=[paramter.batch_size,None],name="decode_inpur")
decode_target=tf.placeholder(tf.int32,shape=[paramter.batch_size,None],name="decode_target")
lstm_network=deep_learning_network.LSTM()



lstm_network.init_train_parameter(paramter)
train,loss,logit=lstm_network.train(encode_input,decode_input,decode_target,paramter.time_major)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i=0
    while True:
        print(i)
        i=i+1
        x,y,z=lstm_data.read_next_batch_data(paramter.batch_size)
        _,train_loss,train_logits=sess.run([train,loss,logit],feed_dict={encode_input:x,decode_input:y,decode_target:z})
        out = train_logits.tolist()

        a=[]
        for item in out[0]:
            a.append(item.index(max(item)))

        print(z[0])
        print(a)




        print(train_loss)

        pass









