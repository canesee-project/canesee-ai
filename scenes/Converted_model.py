from keras.models import Model
# @tf.function
# def train_step(img_tensor, target):
  #loss = 0


#replace target.shape[0] with 64
hidden = tf.zeros((BATCH_SIZE, units))

dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
  
#CNN ENCODER
features = tf.keras.layers.Dense(img_name_train, activation='relu')
#Attention model
          # passing the features through the decoder
hidden_with_time_axis = tf.expand_dims(hidden, 1)
            # score shape == (batch_size, 64, hidden_size)
W1 = tf.keras.layers.Dense(features)
W2 = tf.keras.layers.Dense(hidden_with_time_axis)
score = tf.nn.tanh(W1 + W2)
V = tf.keras.layers.Dense(score)
    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
attention_weights = tf.nn.softmax(V, axis=1)
    # context_vector shape after sum == (batch_size, hidden_size)
context_vector = attention_weights * features
context_vector = tf.reduce_sum(context_vector, axis=1)

#x is dec_input, RNN decoder
x = embedding(dec_input)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    #AB3TLAHA UNITS WALLA DEC_INPUT??
output, state = tf.keras.layers.GRU(x,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')(x)

    # shape == (batch_size, max_length, hidden_size)
x =  tf.keras.layers.Dense(output)

    # x shape == (batch_size * max_length, hidden_size)
x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
outputs= tf.keras.layers.Dense(x)

         # predictions, hidden, _ = decoder(dec_input, features, hidden)
    

  #loss += loss_function(target[:, i], predictions)
  	# tie it together [image, seq] [word]
    #dy mesh akeed!!
model = Model(inputs=[features, dec_input], outputs=outputs)
 
model.compile(loss=tf.keras.losses.SparseCategoricalCrossEntropy(), optimizer='adam')
	  #tf.data.Dataset
model.fit(dataset, epochs=20)

