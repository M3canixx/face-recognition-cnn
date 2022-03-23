import warnings
warnings.filterwarnings("ignore")
from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import ReduceLROnPlateau


def main(input_shape, nombre_classes):
    resnet = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    resnet.tbatch_sizenable = False
    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(nombre_classes, activation='softmax'))
    
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.7, 
                                            min_lr=0.00000000001)
    return model, learning_rate_reduction