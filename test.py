import tensorflow as tf
#---------------------------------------------

# batch_size=3
#
# target_data=[[2,2,2,3,0,0,0],[2,3,0,0,0,0,0],[2,2,2,2,2,3,0]]
#
# x = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
# b=tf.fill([batch_size, 1], 1)
# y = tf.concat([tf.fill([batch_size, 1], 1), target_data], 1)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     output=sess.run(y)
#     print(output)


# #--------------------------------------------------
# batch_data=[[2,2,2,3,0],[2,2,3,0,0]]
#
# y = tf.concat([tf.fill([2, 1], 1), batch_data], 1)
#
# sequence_length = tf.cast(tf.reduce_sum(tf.sign(tf.abs(y)), 1), tf.int32)-1
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     y,output=sess.run([y,sequence_length])
#     print(y)
#     print(output)
#-------------------------------
# data=[1,2,3]
# maxlength=5
# data_type=tf.float32
# y=tf.sequence_mask(data,maxlength,data_type)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     output=sess.run(y)
#     print(output)

#--------------------------------
# import data_process
# chinese_train_data_path="data\wenxue2.txt"
# pinyin_path="data\pinyin.txt"
# encode=data_process.EncodeData_ChineseToPingyin(chinese_train_data_path,pinyin_path)
# encode.data_process()
# encode.decodetest()

# train_pinyin=[[1,2,3],[2,3,4]]
# print(list(map(lambda x:x.append(3),train_pinyin)))
# print(train_pinyin)
# print(list(map(lambda x: x % 2, range(7))))
#
# def f(x):
#     return x*x
# print(map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9]))

#------------------------------------------
# x=[[0.1,0.1,0.8],[0.2,0.5,0.3]]
# mysoftmax=tf.nn.softmax(x)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     out=(sess.run(mysoftmax)).tolist()
#     print(out[1].index(max(out[1])))
#-----------------------------------------------------------
# target_weights = tf.sequence_mask([2,3,2], 5, dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print( sess.run(tf.reduce_sum((target_weights))))

import data_process

mydatap=data_process.EncodeData_ChineseToPingyin(chinese_train_data_path="data\wenxue2.txt",pinyin_path="data\pinyin.txt")
mydatap.decodetest()







