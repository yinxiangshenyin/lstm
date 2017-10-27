import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class LSTM:
    def _embedding(self,embedding_input,input_class_size,embedding_size,):
        '''embedding the input

        :param embedding_size: the firt lstm cell num_units
        :param input_class_size:the input class size
        :param embedding_input: the input of the embedding

        embedding_input=[batch_size,time_squence]
        embedding_inp=[batch_size,time_sequence,embedding_size]

        :return:
        '''
        #embedding_init=tf.get_variable(shape=[input_class_size,embedding_size])
        embedding_init=tf.Variable(tf.truncated_normal(shape=[input_class_size, embedding_size], stddev=0.1),
                    name='encoder_embedding')

        embedding_inp=tf.nn.embedding_lookup(embedding_init,embedding_input)
        return embedding_init,embedding_inp

    def _single_cell(self,unit_type, num_units, mode,dropout=0.0,
                      residual_connection=False, device_str=None):
        """Create an instance of a single RNN cell."""
        # dropout (= 1 - keep_prob) is set to 0 during eval and infer
        dropout = dropout if mode == "train" else 0.0
        # Cell Type
        if unit_type == "lstm":
            single_cell = tf.contrib.rnn.BasicLSTMCell(num_units)

        elif unit_type == "gru":
            single_cell = tf.contrib.rnn.GRUCell(num_units)

        elif unit_type == "layer_norm_lstm":
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units,layer_norm=True)
        else:
            raise ValueError("Unknown unit type %s!" % unit_type)

        # Dropout (= 1 - keep_prob)
        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))

        # Residual
        if residual_connection:
            single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)

        # Device Wrapper
        if device_str:
            single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)

        return single_cell
    def _get_squence_length(self,batch_data):
        ''' get the length squence
        the sequence is padding 0

        eg:
        data:
        {[1,2,3,0,0],
          [1,2,3,4,5]}
        return
         [3,5]

        :param batch_data:
        :return:
        '''
        sequence_length = tf.cast(tf.reduce_sum(tf.sign(tf.abs(batch_data)), 1), tf.int32)
        return sequence_length

    def _build_multiLSTM(self,encode_lstm_layers_num,encode_lstm_cell_num,residual_num_layers,mode,dropout=0.0):

        cell_list = []
        for i in range(encode_lstm_layers_num):
            single_cell = self._single_cell(
                unit_type='lstm',
                num_units=encode_lstm_cell_num,
                mode=mode,
                dropout=dropout,
                residual_connection=(i >= encode_lstm_layers_num - residual_num_layers))
            cell_list.append(single_cell)

        lstm_cell = tf.contrib.rnn.MultiRNNCell(cell_list)

        return lstm_cell

    def _process_decoder_input(self,target_data, tgt_sos_id, batch_size):
        """
        Preprocess target data for decoding
        :param target_data: Target Placehoder
        :param tgt_sos_id: the id of label of  start decoding
        :param batch_size: Batch Size
        :return: Preprocessed target data
        """
        x = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        y = tf.concat([tf.fill([batch_size, 1], tgt_sos_id), x], 1)

        for item in y:
            item.remove(3)
        return y

    def _compute_loss(self, target,logits,batch_size,time_major):
        """Compute optimization loss."""
        target_output = target
        max_time =tf.shape(target_output)[1]#len(target_output[0])
        target_sequence_length = self._get_squence_length(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(target_sequence_length, max_time, dtype=logits.dtype)
        # if time_major:
        #     target_weights = tf.transpose(target_weights)
        loss = tf.reduce_sum(crossent * target_weights) /tf.to_float(batch_size) #tf.to_float(tf.reduce_sum(target_weights))
        return loss


    def _build_encode_lstm(self,encode_input,encode_input_class_length,encode_lstm_cell_num,encode_lstm_layers_num,
                    mode,dropout=0.0,residual_num_layers=0,time_major=False):
        '''design the encode part of lstm

        :param encode_input: the input of the encode
        :param encode_input_class_length:  the class number of the input
        :param encode_lstm_cell_num: the layer number of the lstm cell
        :param encode_lstm_layers_num: the layer number of the lstm
        :param mode: 'train' 'test' which is import for droupout
        :param dropout: discard the cell_nunits percent
        :param residual_num_layers:
        :return:
        '''
        _,encoder_emb_inp=self._embedding(encode_input,encode_input_class_length,encode_lstm_cell_num)
        input_sequence_length=self._get_squence_length(encode_input)

        lstm=self._build_multiLSTM(encode_lstm_layers_num,encode_lstm_cell_num,residual_num_layers,mode,dropout)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            lstm,
            encoder_emb_inp,
            dtype=tf.float32,
            sequence_length=input_sequence_length,
            time_major=time_major)

        return encoder_outputs, encoder_state

    def _build_decoder_lstm(self, decode_input,batch_size,encoder_state,
                            decode_input_class_length, tgt_sos_id,tgt_eos_id,
                            decode_lstm_cell_num, decode_lstm_layers_num,
                            mode,maximum_iterations,dropout=0.5, residual_num_layers=0, time_major=False,
                            ):
        '''Build an RNN cell that can be used by decoder.

        :param decode_input:
        :param encoder_state:
        :param decode_input_class_length:
        :param decode_lstm_cell_num:
        :param decode_lstm_layers_num:
        :param mode:
        :param dropout:
        :param residual_num_layers:
        :param time_major:
        :return:
        '''
        dropout = 0.0 if mode != "train" else dropout

        # decode_input = self._process_decoder_input(decode_input, tgt_sos_id, batch_size)
        #decode_input = self._process_decoder_input(decode_input, tgt_sos_id, batch_size)
        lstm = self._build_multiLSTM(decode_lstm_layers_num, decode_lstm_cell_num, residual_num_layers, mode, dropout)
        output_layer = layers_core.Dense(decode_input_class_length, use_bias=False, name="output_projection")
        decoder_emb, decoder_emb_inp = self._embedding(decode_input, decode_input_class_length, decode_lstm_cell_num)
        target_sequence_length = self._get_squence_length(decode_input)


        outputs=None
        if mode=="train":
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,target_sequence_length,time_major=time_major)
            # Decoder
            my_decoder = tf.contrib.seq2seq.BasicDecoder(lstm, helper,encoder_state, output_layer=output_layer )
            #dynamic_decode
            outputs, final_context_state,_= tf.contrib.seq2seq.dynamic_decode(my_decoder,maximum_iterations=maximum_iterations,output_time_major=time_major,swap_memory=True)

        if mode=="Inference":
            start_tokens = tf.fill([batch_size], tgt_sos_id)
            end_token = tgt_eos_id
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_emb, start_tokens, end_token)
            my_decoder = tf.contrib.seq2seq.BasicDecoder(lstm, helper,encoder_state,output_layer=output_layer)
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode( my_decoder,maximum_iterations=maximum_iterations,
                                                                                 output_time_major=time_major,swap_memory=True)
        return outputs

    def _seq2seq_model(self,encode_input,encode_input_class_length,encode_lstm_cell_num,encode_lstm_layers_num,
                      decode_input,decode_input_class_length,decode_lstm_cell_num,decode_lstm_layers_num,tgt_sos_id,tgt_eos_id,
                      mode,batch_size,dropout=0.5,residual_num_layers=0,time_major=False):

        dropout=0.0 if mode!="train" else dropout

        encoder_outputs, encoder_state=self._build_encode_lstm(encode_input,encode_input_class_length,encode_lstm_cell_num,encode_lstm_layers_num,
                                                               mode,dropout=dropout,residual_num_layers=residual_num_layers,time_major=time_major)

        mymaximum_iterations=tf.shape(encode_input)[1]#len(encode_input[0])

        output=self._build_decoder_lstm(decode_input, batch_size, encoder_state,decode_input_class_length, tgt_sos_id, tgt_eos_id,
                                        decode_lstm_cell_num, decode_lstm_layers_num, mode,maximum_iterations=mymaximum_iterations,dropout=dropout,
                                        residual_num_layers=residual_num_layers, time_major=time_major)

        return output

    def init_train_parameter(self,lstm_paramter):


        self.encode_input_class=lstm_paramter.encode_input_class
        self.encode_input_class_length=len(lstm_paramter.encode_input_class)
        self.encode_lstm_cell_num=lstm_paramter.encode_lstm_cell_num
        self.encode_lstm_layers_num=lstm_paramter.encode_lstm_layers_num

        self.decode_input_class=lstm_paramter.decode_input_class
        self.decode_input_class_length=len(lstm_paramter.decode_input_class)
        self.decode_lstm_cell_num=lstm_paramter.decode_lstm_cell_num
        self.decode_lstm_layers_num=lstm_paramter.decode_lstm_layers_num
        self.tgt_sos_id=lstm_paramter.tgt_sos_id
        self.tgt_eos_id=lstm_paramter.tgt_eos_id


        self.batch_size=lstm_paramter.batch_size
        self.dropout=lstm_paramter.dropout
        self.residual_num_layers=lstm_paramter.residual_num_layers





    def train(self,encode_input,decode_input,decode_target,time_major=False):
        mode="train"
        self.time_major=time_major
        if self.time_major:
            encode_input=tf.transpose(encode_input)
            decode_input=tf.transpose(decode_input)
            decode_target=tf.transpose(decode_target)


        output=self._seq2seq_model(encode_input,self.encode_input_class_length,self.encode_lstm_cell_num,self.encode_lstm_layers_num,
                           decode_input,self.decode_input_class_length,self.decode_lstm_cell_num,self.decode_lstm_layers_num,self.tgt_sos_id,self.tgt_eos_id,
                            mode,self.batch_size,dropout=self.dropout,residual_num_layers=self.residual_num_layers,time_major=self.time_major)

        train_loss=self._compute_loss(decode_target,output.rnn_output,self.batch_size,self.time_major)
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        grads_and_vars = opt.compute_gradients(train_loss,params)

        clipped_grads_and_vars = [(tf.clip_by_norm(grad,5), var) for grad, var in grads_and_vars]

        train_op=opt.apply_gradients(clipped_grads_and_vars)

        logit=output.rnn_output


        return train_op,train_loss,logit
        # return output.rnn_output,decode_input





















