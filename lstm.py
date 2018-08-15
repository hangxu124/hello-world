from keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D, TimeDistributed,Convolution2D,Activation,MaxPool2D,Flatten,LSTM
from keras.applications import vgg16
from keras.models import Model,Sequential
from keras.utils import plot_model
from keras import backend as K
K.set_image_dim_ordering('tf')
'''
vis = Input(shape=(100,200,200,3))
base = vgg16.VGG16
base =base(include_top=False, input_shape=(200,200,3))
x=base.output
x=GlobalAveragePooling2D()(x)
x=Dropout(0.4)(x)
x=Dense(1024,activation='relu')(x)
p=Dense(2,activation='softmax')(x)
model=Model(input=base.input,output=p)
print (model.summary())
#plot_model(model,to_file="model.png")

model=Sequential()
model.add(TimeDistributed(Convolution2D(32,3,3,border_mode="valid"),input_shape=(10,1,200,200)))
convout1=Activation('relu')
model.add(TimeDistributed(convout1))
model.add(TimeDistributed(Convolution2D(32,3,3)))
convout2=Activation('relu')
model.add(TimeDistributed(convout2))
model.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
model.add(TimeDistributed(Dropout(0.32)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(20,return_sequences=True))
model.add(TimeDistributed(Dense(128)))
model.add(TimeDistributed(Dense(2)))
model.add(TimeDistributed(Activation('softmax')))
print(model.summary())
'''
# model=Sequential()
# model.add(Convolution2D(32,3,3,border_mode="valid"))
# convout1=Activation('relu')
# model.add(convout1)
# model.add(Convolution2D(32,3,3))
# convout2=Activation('relu')
# model.add(convout2)
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.32))
# model.add(Flatten())
# model.add(LSTM(20,return_sequences=True))
# model.add(Dense(128))
# model.add(Dense(2))
# model.add(Activation('softmax'))
# print(model.summary())

# cnn=vgg16.VGG16(include_top=False,pooling='avg')
# cnn.trainable=False
# print (cnn.inputs)
# H=W=200
# C=3
# video_input=Input(shape=(None,H,W,C),name='video_input')
# encodedframe=TimeDistributed(cnn)(video_input)
# encodedvideo=LSTM(20)(encodedframe)
# op=Dense(256,activation='relu')(encodedvideo)
# videomodel=Model(inputs=[video_input],outputs=op)
# print(videomodel.summary())


import tensorflow as tf

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.applications import InceptionV3, VGG19
from keras.layers import TimeDistributed

import numpy as np

def main():
    ## Define vision model
    ## Inception (currently doesn't work)
    #cnn = InceptionV3(weights='imagenet',
    #                  include_top='False',
    #                  pooling='avg')

    # Works
    cnn = VGG19(weights='imagenet',
                include_top='False', pooling='avg')

    cnn.trainable = False

    H=W=229
    C = 3
    video_input = Input(shape=(None,H,W,C), name='video_input')

    encoded_frame_sequence = TimeDistributed(cnn)(video_input) # the output will be a sequence of vectors

    encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

    output = Dense(2, activation='relu')(encoded_video)

    video_model = Model(inputs=[video_input], outputs=output)

    print(video_model.summary())

    video_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    #video_model.compile(optimizer='adam', loss='mean_squared_error')

    #features = np.empty((0,1000))

    n_samples = 20
    n_frames = 10

    frame_sequence = np.random.randint(0.0,255.0,size=(n_samples, n_frames, H,W,C))

    y = np.random.random(size=(2,20))
    y = np.reshape(y,(-1,2))

    print(frame_sequence.shape)

    video_model.fit(frame_sequence, y, nb_epoch=10, validation_split=0.0,shuffle=False, batch_size=1)

if __name__=='__main__':
    main()