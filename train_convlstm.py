from config import Config
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2, os, math, numpy as np


def get_frames(video_path):
    capture = cv2.VideoCapture(video_path)

    # count the number of frames
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

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

    x = math.ceil((len(resized_frames) - Config.window)/Config.step)
    necessary = Config.window + x * Config.step

    # copy first frame and put this copies at the beggining
    for _ in range(necessary - len(resized_frames)):
        resized_frames.insert(0, np.copy(resized_frames[0]))

    # yield parts (windows)
    part = 0
    while True:
        if part * Config.step + Config.window == len(resized_frames):
            break
        yield np.asarray(resized_frames[part * Config.step : part * Config.step + Config.window])
        part += 1


def create_data(paths, y):
    x_resized, y_onehot = [], []
    
    for vid_file in paths:
        for part in get_frames(vid_file):
            assert part.shape == (Config.window, Config.img_dim, Config.img_dim, 3)
            x_resized.append(part)
            y_onehot.append(np.asarray(y[paths.index(vid_file)], dtype=np.uint8))

    x_resized = np.asarray(x_resized)
    y_onehot = np.asarray(y_onehot)

    return x_resized, y_onehot


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
    opt = Adam(learning_rate=0.0001, lr=0.0001, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    if not os.path.exists(Config.checkpoint_path):
        os.makedirs(Config.checkpoint_path)

    cp_callback = ModelCheckpoint(filepath=Config.checkpoint_path, save_weights_only=True, verbose=1)

    earlystop = EarlyStopping(patience=7)
    callbacks = [cp_callback, earlystop]

    model.fit(x=x_train, y=y_train, epochs=40, batch_size=8, shuffle=True, validation_split=0.2, callbacks=callbacks)

    if not os.path.exists(Config.data_path):
        os.makedirs(Config.data_path)

    model.save(Config.data_path + 'model')
    model.save(Config.data_path + 'model.h5')
    model.save_weights(Config.data_path + 'weights')

    return model


def predict(x_test):
    model = load_model(Config.data_path + 'model.h5')

    return model.predict(x_test)


def save(**kwargs):
    print('Saving...')
    if not os.path.exists(Config.convlstm_train_dataset_path):
        os.makedirs(Config.convlstm_train_dataset_path)
    
    for arr in kwargs.keys():
        np.save(os.path.join(Config.convlstm_train_dataset_path, arr + '.npy'), kwargs[arr])

    print('\nDone.')


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


def split():
    y_classes = os.listdir(Config.dataset_path)
    x_train_test_paths = {'train': [], 'test': []}
    y_train_test = {'train': [], 'test': []}

    for cl in y_classes:
        act_files = os.listdir(os.path.join(Config.dataset_path, cl))
        groups = {}
        for vid_file in act_files:
            vid_path = os.path.join(os.path.join(Config.dataset_path, cl), vid_file)
            vid_path_tokens = vid_path.split('_')
            path_without_index = '_'.join(vid_path_tokens[:len(vid_path_tokens) - 1])

            if not groups.get(path_without_index):
                groups[path_without_index] = [vid_path]
            else:
                groups[path_without_index].append(vid_path)

        group_len = [(g, len(groups[g])) for g in groups.keys()]
        sorted_group_len = sorted(group_len, key=lambda x: x[1])

        added_into_test = 0
        for group, _ in sorted_group_len:
            paths = [path for path in groups[group]]

            y = [0] * len(y_classes)
            y[y_classes.index(cl)] = 1

            if added_into_test < int((1 - Config.train_part) * len(act_files)):
                [x_train_test_paths['test'].append(path) for path in paths]
                [y_train_test['test'].append(k) for k in [y] * len(paths)]
                added_into_test += len(paths)
            else:
                [x_train_test_paths['train'].append(path) for path in paths]
                [y_train_test['train'].append(k) for k in [y] * len(paths)]

    return x_train_test_paths['train'], x_train_test_paths['test'], y_train_test['train'], y_train_test['test']


if __name__ == '__main__':
    # --------- Create and save data ---------
    x_train_paths, x_test_paths, y_train, y_test = split()

    x_train, y_train = create_data(x_train_paths, y_train)
    x_test, y_test = create_data(x_test_paths, y_test)
    save(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, classes=np.asarray(os.listdir(Config.dataset_path)))
    
    # --------- Load data and train model ---------
    # data = load()
    # x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    
    # model = train_model(x_train, y_train)
