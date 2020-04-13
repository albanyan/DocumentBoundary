from config import window_size
from keras.models import Sequential, Model
from keras.layers import Input, Dense,concatenate, Dropout,LeakyReLU,Conv2D,MaxPooling2D,Dropout,Flatten,TimeDistributed,LSTM
import numpy as np
def build_model():
    InLSTM = Input(shape=(window_size,100,100,1))
    ConvLSTM1 = TimeDistributed(Conv2D(32, kernel_size=(7,7),activation='relu'),name="Conv2d_1")(InLSTM)
    PoolLSTM1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)),name="MaxPooling2D_1")(ConvLSTM1)
    DropoutLSTM1 = TimeDistributed(Dropout(0.25),name="Dropout_1")(PoolLSTM1)
    ConvLSTM2 = TimeDistributed(Conv2D(32, kernel_size=(5,5),activation='relu'),name="Conv2d_2")(DropoutLSTM1)
    PoolLSTM2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)),name="MaxPooling2D_2")(ConvLSTM2)
    DropoutLSTM2 = TimeDistributed(Dropout(0.5),name="Dropout_2")(PoolLSTM2)
    FlatLSTM = TimeDistributed(Flatten(),name="Flatten_1")(DropoutLSTM2)
    Lstm = LSTM(32, activation='relu',dropout=0.5)(FlatLSTM)
    OutLSTM = Dense(2,activation='softmax') (Lstm)


    InCNN = Input(shape=(100,100,1))
    Conv1 = Conv2D(32, kernel_size=(11,11),activation='relu')(InCNN)
    Pool1=MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = Conv2D(64, (5, 5), activation='relu')(Conv1)
    Pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Drop = Dropout(0.5)(Pool2)
    Flat = Flatten()(Drop)
    Dense1 = Dense(128, activation='relu')(Flat)
    DenseDrop = Dropout(0.5)(Dense1)
    OutCNN = Dense(2, activation='softmax')(DenseDrop)


    LstmCnn = concatenate([Lstm, Flat])
    densef = Dense(32,activation="relu")(LstmCnn)
    dropf = Dropout(0.5)(densef)
    OutMainModel = Dense(2,activation="softmax")(dropf)
    MainModel = Model(inputs=[InLSTM,InCNN], outputs=OutMainModel)

    return MainModel

def detect(model,model_path,X_testLSTM,X_testCNN):
    model.load_weights(model_path)
    EOD_cat = model.predict([X_testLSTM,X_testCNN])
    EOD = np.argmax(EOD_cat, axis=1)
    return EOD
