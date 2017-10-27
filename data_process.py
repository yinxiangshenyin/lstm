import re
from collections import Counter
import os
import pinyin
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag
import numpy as np
import pickle
import copy
import tensorflow as tf

class General_Operation(object):
    '''the gerneral operation of data processing

    '''

    def file_to_list(self, filename):
        '''load data from the file and channg the data to list by punctuation mark then rewrite to the file

        :param filename: the file which content is not in the list style
        :return: None
        '''
        content_list = []
        # get the content from the file and segment it
        with open(filename, 'r', encoding='utf-8') as  readfile:
            file = readfile.read()
            file = file.strip('\n').strip()
            file = re.sub(r'[A-Za-z0-9]|/d+', '', file)
            file = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——? -:；.、”“ ：<> 《》~@#￥%……&*（）]+", "",
                          file)  # eliminate non-chinese characters
            content_list = re.split(" |！|\？|\。|\，", file)  # segment the content by the punctuation mark

        # rewrite to the file
        fobj = open(filename, 'w', encoding="utf-8")
        fobj.writelines(('%s%s' % (str(items), os.linesep) for items in content_list))  # content是要保存的list。
        fobj.close()
        return file

    def read_orignal_data(self, filename):
        '''read the original data form the file
        the content of the file must be segmentation by single row
        and not has been encoded

        :param filename: file path
        :return:
        '''
        file = []
        with open(filename, 'r', encoding='utf-8') as readfile:
            for item in readfile:
                item = item.strip('\n').strip()
                if (len(item) > 1):
                    file.append(item)
        return file

    def get_class(self,data,frequence_limit=0,start_index=0):
        ''' get the dictionaty from the data

        :param data: the data which will be statistics
        :param limit:  the limit of the frequence to count
        :param start_index:the start index of dictionary
        :return:
        '''
        all_word=[]
        for item in data:
            all_word+=([word for word in item])          # count all the words
        counter=Counter(all_word)    #count the number of the occurrences of the words return the style by dict
        counter_copy=copy.deepcopy(counter)

        for key,value in counter_copy.items():          #del the word  frequence less 3
            if value<=frequence_limit:
                del counter[key]

        counter_pair = sorted(counter.items(), key=lambda x: x[1])     #list of tuple
        word_class,_=zip(*counter_pair)         #unzip the dictionary
        word_class_dic=dict(zip(word_class,range(start_index,len(word_class)+start_index)))         #redictionary the words by 2-length+2
        return word_class_dic

    def encode(self,data, dictionary):
        '''encode the date by the dictionary

        :param data:
        :param dictionary:
        :return:
        '''
        train_label = []
        for row in data:
            x = []
            for item in row:
                x.append(dictionary.get(item, 1))           #encode the data to numby and 1 in the dictionary represent "UNK"
            train_label.append(x)
        return train_label

    def decode(self,data, dictionary):
        '''decode the data by the dictionary
        the dictionary is same with the encode's dictionary

        :param labels:
        :param label_class_dict:
        :return:
        '''
        decode_dict = dict((value, key) for key, value in dictionary.items())
        encode_label = []
        for label in data:
            x = ""
            for item in label:
                x = x + (decode_dict.get(int(item), ' '))+" "            #decode the data and append " " between the item and if not in deictionary will retuen " "
            encode_label.append(x)
        return encode_label

    def load_data_pick(self,filename):
        '''load the apk data which has been encoded to numpy

        :param filename:
        :return:
        '''
        f=open(filename,'rb')
        data=pickle.load(f)
        return data

    def save_data_pick(self,filemane,data):
        '''save the encoded data to pick file

        :param filemane: target filename
        :param data:
        :return:
        '''
        f = open(filemane, 'wb')
        pickle.dump(data, f, 0)
        f.close()

    def get_max_len_from_list(self, list):
        '''get the max len of the list data

        :param list: list of data
        :return:
        '''
        return np.max([len(item) for item in list])

    def one_hot_encode(self,data,class_length):
        ''' one hot encoding

        :param data: list of data
        :param class_length: length of the dictionary class

        :return:
        '''
        max_length_of_data=self.get_max_len_from_list(data)
        sequence=np.zeros((len(data),max_length_of_data,class_length))
        for i,sentence in enumerate(data):
            for j,number in enumerate(sentence):
                sequence[i,j,number]=1
        return sequence

class EncodeData_ChineseToPingyin(General_Operation):
    ''' this class is to encode the train data on chinese and the labels on pinyin that are responding to train data
    ----------------------------------------------
    usage:
    mydata=data_process.EncodeData_ChineseToPingyin(chinese_train_data_path=r"data\wenxue2.txt", pinyin_path=r"data\pinyin.txt")
    mydata.data_process()
    mydata.decodetest()
    -----------------------------------------------
    train data:
    the class need the path of the train data ,which is aboat chinese

    pinyin:
    the combination of the pinyin which contain  40~50 items

    target label:
    the target labels is pinyin labels which gets form the train data by the function of this class

    chinese dictionary:
    the class need the the dict of chinese which to encode the chinese to list of numby
    if not the chinese dictionary it will learn the dictionary with the train data

    pinyin dictionary：
    the class need the dict of the pinyin to encode the pinyin to the list of numby
    if not the pinyin dictionary it will get the dictionary with the label

    encode the data and the labels by number
    on top of that but the label need encode to one-hot encode

    '''

    def __init__(self,chinese_train_data_path,pinyin_path,chinese_dictionary_path=None,pinyin_dictionary_path=None,
                 save_encode_train_data_path='data\chinese_train.txt',save_encode_target_data_path='data\pinyin_target.txt',
                 save_chinese_dictionary_path='data\chinese_dictionary.txt',save_pinyin_dictionary_path='data\pinyin_dictionary.txt'):
        '''the init of the EncodeData_chinese_pinyin

        Args:
            chinese_train_data_path:
                the file path of the chinese train data
            pinyin_path:
                the file path of the pinyin file which contains the combination of the pinyin ,if the pinyin dictionary is not None this parameter will no meaning
            chinese_dictionary_path:
                the file path of the chinese dictionary
            pinyin_dictionary_path:
                the file path of the pinyin dictionary
            save_encode_train_data_path:
                the path to save the train data which has been encoded
            save_encode_target_path:
                the path to save the traget lable which has benn encoded
            save_chinese_dictionary_path:
                the path to save the chinese dictionary ,if the 'chinese_dictionary_path' is not None ,this parmater will no meaning
                so that will not resave the chinese dictionary
            save_pinyin_dictionary_path:
                the path to save the pinyin dictionary ,if the 'pinyin_dictionary_path' is not None ,this parmater will no meaning
                so the will no resave the pinyin dictionary
        '''

        self.chinese_train_data_path=chinese_train_data_path
        self.pinyin_path=pinyin_path
        self.chinese_dictionary_path=chinese_dictionary_path
        self.pinyin_dictionary_path=pinyin_dictionary_path
        self.save_encode_train_data_path=save_encode_train_data_path
        self.save_encode_target_data_path=save_encode_target_data_path
        self.save_chinese_dictionary_path=save_chinese_dictionary_path
        self.save_pinyin_dictionary_path=save_pinyin_dictionary_path

        self.mypin = pinyin     #the third module to trainslate the chinese to pinyin

    def chinese_to_pinyin(self,data):
        '''get the pinyin sequence from the chinese sequence

        :param data: the list of chinese data
        :return: pinyin list
        '''
        pinyinlist = []
        for item in data:
            x=[]
            item=item.strip('')
            for i in item:
                pinyin=self.mypin.get(i, format="strip", delimiter=" ")         #get the pinyin of chinese
                if pinyin==i:
                    x.append("UNK")
                else:
                    x.append(pinyin)
            pinyinlist.append(x)
        return pinyinlist

    def pinyin_to_chinese(self,data):
        '''get the chinese from the pinyin

        :param data: pinyin data
        :return:
        '''
        dagparames = DefaultDagParams()
        result = dag(dagparames, data, path_num=10, log=True)
        for item in result:
            print(str(item.score) + ":", item.path)

    def data_process(self):
        #read the oringinal data from the file
        #self.file_to_list(self.chinese_train_data_path)
        train_data=self.read_orignal_data(self.chinese_train_data_path)
        #get the chinese dictionary
        if self.chinese_dictionary_path !=None:
            self.chinese_encode_dict=self.load_data_pick(self.chinese_dictionary_path)
        else:
            self.chinese_encode_dict = self.get_class(train_data,frequence_limit=3,start_index=4)
            # set the dictionary 0-zero 1-unk
            self.chinese_encode_dict['ZERO'] = 0
            self.chinese_encode_dict['UNK'] = 1
            self.chinese_encode_dict['START']=2
            self.chinese_encode_dict['STOP'] =3

        #get the pinyin dictionary
        if self.pinyin_dictionary_path !=None:
            self.pinyin_encode_dict = self.load_data_pick(self.pinyin_dictionary_path)
        else:
            pinyin_data=self.read_orignal_data(self.pinyin_path)
            pinyin = []
            for item in pinyin_data:
                pinyin.append(item.split(' '))
            self.pinyin_encode_dict = self.get_class(pinyin,frequence_limit=0,start_index=2)
            # set the dictionary 0-zero 1-unk
            self.pinyin_encode_dict['zero'] = 0
            self.pinyin_encode_dict['UNK'] = 1
            self.pinyin_encode_dict['START'] = 2
            self.pinyin_encode_dict['STOP'] = 3
        print(sorted(self.pinyin_encode_dict.items(),key=lambda x:x[1]))
        print(sorted(self.chinese_encode_dict.items(),key=lambda x:x[1]))
        #get the pinyin from the data
        #train_data.sort(key=lambda x:len(x))
        train_pinyin=self.chinese_to_pinyin(train_data)
        #encode the train and target data to numpy
        self.train_pinyin=self.encode(train_pinyin,self.pinyin_encode_dict)
        #list(map(lambda x: x.append(3), self.train_pinyin) )   #add the stop to each sequence
        self.train_data = self.encode(train_data, self.chinese_encode_dict)
        #list(map(lambda x: x.append(3), self.train_data))
        #get the max length of data
        self.train_pinyin_max_len = self.get_max_len_from_list(self.train_pinyin)
        self.train_data_max_len = self.get_max_len_from_list(self.train_data)
        #order the train data
        self.train_data=sorted(self.train_data,key=lambda x:len(x))
        self.train_pinyin=sorted(self.train_pinyin,key=lambda x:len(x))
        #---------------------------------------------------------------------
        #save the data
        #not need very time
        self.save_data_pick(self.save_encode_train_data_path,self.train_data)
        self.save_data_pick(self.save_encode_target_data_path, self.train_pinyin)
        self.save_data_pick(self.save_chinese_dictionary_path, self.chinese_encode_dict)
        self.save_data_pick(self.save_pinyin_dictionary_path, self.pinyin_encode_dict)
        #----------------------------------------------------------------------

    def decodetest(self,save_encode_train_data_path='data\chinese_train.txt',save_encode_target_data_path='data\pinyin_target.txt',
                 save_chinese_dictionary_path='data\chinese_dictionary.txt',save_pinyin_dictionary_path='data\pinyin_dictionary.txt'):
        #load encode data
        encode_train_data=self.load_data_pick(save_encode_train_data_path)
        encode_pinyin_data=self.load_data_pick(save_encode_target_data_path)
        #load dictionary
        chinese_dict=self.load_data_pick(save_chinese_dictionary_path)
        pinyin_dict = self.load_data_pick(save_pinyin_dictionary_path)
        # #verse the dictionary
        # decode_chinese_dict=dict((value,key) for key,value in chinese_dict.items())
        # decode_pinyin_dict=dict((value,key) for key,value in pinyin_dict.items())
        #decode the data
        print(self.decode(encode_train_data[0:9],chinese_dict))
        print(self.decode(encode_pinyin_data[0:9],pinyin_dict))

class ReadDate_LSTM(General_Operation):
    ''' read data for lstm
    before read the data must excuting the EncodeData_ChineseToPingyin class
    -----------------------------------------------------------
    usage:
    mydata=data_process.ReadDate_LSTM(input_data_path='data\chinese_train.txt',target_data_path='data\pinyin_target.txt',
                                  input_data_dict_path='data\chinese_dictionary.txt',target_data_dict_path='data\pinyin_dictionary.txt')
    while  True:
        input,target=mydata.read_next_batch_data(3)
    -------------------------------------------------------------
    '''
    def __init__(self,input_data_path,target_data_path,input_data_dict_path,target_data_dict_path):
        ''' the init for the read data for lstm

        :param input_data_path: the file path of the source data,which has been encoded to numby
        :param target_data_path: the file path of the target data which has been encoded to numby
        :param input_data_dict_path: the dictionary of the source data
        :param target_data_dict_path: the dictionary of the target data
        '''

        self.input_data_path=input_data_path
        self.target_data_path = target_data_path
        self.input_data_dict_path = input_data_dict_path
        self.target_data_dict_path = target_data_dict_path
        self.load_data()

    def load_data(self):
        '''the data process of read data for the lstm
        because of the train data is litter so take the measure
        of loading all train data
        :return:
        '''
        self.input_data=np.array(self.load_data_pick(self.input_data_path))
        self.target_data=np.array(self.load_data_pick(self.target_data_path))
        self.input_data_dict=self.load_data_pick(self.input_data_dict_path)
        self.target_data_dict=self.load_data_pick(self.target_data_dict_path)
        self.data_length=len(self.input_data)           #length of the train data
        self.pointer=0          #the point for batch_read train data

    def read_next_batch_data(self,batch_size,time_major=False):
        ''' read batch data from the train data

        :param batch_size:
        :return:[batch,time_sequence]
        '''
        #check if the read point more than the data length
        if self.pointer+batch_size>self.data_length:
            print(self.pointer)
        batch_input=np.array(self.input_data[self.pointer:self.pointer+batch_size])
        batch_target=np.array(self.target_data[self.pointer:self.pointer+batch_size])
        self.pointer=self.pointer+batch_size

        encode_input=batch_input
        list(map(lambda x: x.append(3), encode_input))


        decode_input=copy.deepcopy(batch_target)
        list(map(lambda x: x.insert(0,2), decode_input))

        decode_target=copy.deepcopy(batch_target)
        list(map(lambda x: x.append(3), decode_target))



        batch_input_max_len = self.get_max_len_from_list(encode_input)
        batch_target_max_len = self.get_max_len_from_list(decode_target)

        sequence_encode_input = np.zeros((len(encode_input), batch_input_max_len), dtype=np.int32)
        sequence_decode_input = np.zeros((len(decode_input), batch_target_max_len), dtype=np.int32)
        sequence_decode_target = np.zeros((len(decode_target), batch_target_max_len), dtype=np.int32)

        for i, sentence in enumerate(encode_input):
            for j, number in enumerate(sentence):
                sequence_encode_input[i, j] = number

        for i, sentence in enumerate(decode_input):
            for j, number in enumerate(sentence):
                sequence_decode_input[i, j] = number

        for i, sentence in enumerate(decode_target):
            for j, number in enumerate(sentence):
                sequence_decode_target[i, j] = number





        # #one-hot encoding
        # batch_wav=self.one_hot_encode(batch_input,len(self.input_data_dict))
        # batch_label=self.one_hot_encode(batch_target,len(self.target_data_dict))
        # the batch_input and batch_target is not time_major
        return sequence_encode_input,sequence_decode_input,sequence_decode_target












