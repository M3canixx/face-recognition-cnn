from keras.models import model_from_json

def main(fileName, data_path):
    path = data_path + fileName + ".json"
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    path_weights = data_path + fileName + ".h5"
    loaded_model.load_weights(path_weights)
    loaded_model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=["accuracy"])
    return loaded_model