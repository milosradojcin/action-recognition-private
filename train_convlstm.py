from config import Config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import cv2, os, math, numpy as np


def get_frames(video_path):
    capture = cv2.VideoCapture(video_path)

    # count the number of frames
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ceil = math.ceil(frames/fps)

    # calculate duration of the video
    step = int(fps / Config.fps_to_take)
    index = 0

    resized_frames = []
    while True:
        success, image = capture.read()
        if success:
            if index % step == 0:
                resized_frames.append(cv2.resize(image, (Config.img_dim, Config.img_dim), interpolation=cv2.INTER_AREA))
            index += 1
        else:
            capture.release()
            break

    necessary = (duration_ceil + 1) * step - Config.window

    # copy first frame and put this copies at the beggining
    for _ in range(necessary - len(resized_frames)):
        resized_frames.insert(0, np.copy(resized_frames[0]))

    # yield parts (windows)
    part = 0
    while True:
        if part * Config.step + Config.window > len(resized_frames):
            break
        yield np.asarray(resized_frames[part * Config.step : part * Config.step + Config.window])
        part += 1


def create_data(dir_path):
    x_resized, y_onehot, y_classes = [], [], []

    y_classes = os.listdir(dir_path)
    for cl in y_classes:
        act_files = os.listdir(os.path.join(dir_path, cl))
        for vid_file in act_files:
            for part in get_frames(os.path.join(os.path.join(dir_path, cl), vid_file)):
                assert part.shape == (Config.window, Config.img_dim, Config.img_dim, 3)
                x_resized.append(part)

                y = [0] * len(y_classes)
                y[y_classes.index(cl)] = 1
                y_onehot.append(np.asarray(y, dtype=np.uint8))

        print(f'Class {(y_classes.index(cl) + 1)}/{len(y_classes)}...')

    x_resized = np.asarray(x_resized)
    y_onehot, y_classes = np.asarray(y_onehot), np.asarray(y_classes)

    return x_resized, y_onehot, y_classes


def get_model(class_num):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False, data_format="channels_last",
                         input_shape=(Config.window, Config.img_dim, Config.img_dim, 3)))
    model.add(BatchNormalization())
    # play with layers: batch normalization, multiple ConvLSTM2D layers
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(class_num, activation="softmax"))

    return model


def train_model(x_train, y_train):
    model = get_model(class_num=len(y_train[0]))
    # opt = SGD(lr=0.001)
    opt = Adam(learning_rate=0.001, lr=0.001, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    earlystop = EarlyStopping(patience=7)
    callbacks = [earlystop]

    model.fit(x=x_train, y=y_train, epochs=40, batch_size=8, shuffle=True, validation_split=0.2, callbacks=callbacks)

    return model


def predict(model, x_test):
    return model.predict(x_test)


def save(**kwargs):
    print('Saving...')
    if not os.path.exists(Config.convlstm_train_dataset_path):
        os.makedirs(Config.convlstm_train_dataset_path)
    
    for arr in kwargs.keys():
        np.save(os.path.join(Config.convlstm_train_dataset_path, arr + '.npy'), kwargs[arr])


def load():
    if not os.path.exists(Config.convlstm_train_dataset_path):
        print('Given path does not exists')
        return

    matrices = {}

    for _, _, files in os.walk(Config.convlstm_train_dataset_path):
        for file in files:
            if file.endswith(".npy"):
                matrices[file.split('.')[0]] = np.load(os.path.join(Config.convlstm_train_dataset_path, file), allow_pickle=True)

    return matrices


if __name__ == '__main__':
    # Create and save data
    # x_resized, y_onehot, y_classes = create_data(Config.dataset_path)
    # save(x_resized=x_resized, y_onehot=y_onehot, y_classes=y_classes)
    
    # Load data
    train_data = load()
    
    x_resized = train_data['x_resized']
    y_onehot, y_classes = train_data['y_onehot'], train_data['y_classes']

    # Split data and train model
    x_train, x_test, y_train, y_test = train_test_split(x_resized, y_onehot, test_size=0.20, shuffle=True, random_state=0)
    
    model = train_model(x_train, y_train)
