import sys
from typing import Generator, List, Tuple

from numpy.testing._private.nosetester import NoseTester
from config import Config
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cv2, os, math, numpy as np
import typer

typer_app = typer.Typer()


# --------------- PREPARATION ---------------

def split() -> Tuple[list, list, list, list]:
    if not os.path.exists(Config.dataset_path):
        print('Your dataset is not in', os.path.abspath(Config.dataset_path))
        sys.exit(0)

    y_classes = os.listdir(Config.dataset_path)
    x_train_test_paths = {'train': [], 'test': []}
    y_train_test = {'train': [], 'test': []}

    for index, cl in enumerate(y_classes):
        act_path = os.path.join(Config.dataset_path, cl)
        act_files = os.listdir(act_path)
        groups = {}
        for vid_file in act_files:
            vid_path = os.path.abspath(os.path.join(act_path, vid_file))
            vid_path_tokens = vid_path.split('_')
            path_without_index = '_'.join(vid_path_tokens[:len(vid_path_tokens) - 6])

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
            y[index] = 1

            if added_into_test < int((1 - Config.train_part) * len(act_files)):
                [x_train_test_paths['test'].append(path) for path in paths]
                [y_train_test['test'].append(k) for k in [y] * len(paths)]
                added_into_test += len(paths)
            else:
                [x_train_test_paths['train'].append(path) for path in paths]
                [y_train_test['train'].append(k) for k in [y] * len(paths)]

    return x_train_test_paths['train'], x_train_test_paths['test'], y_train_test['train'], y_train_test['test']


def get_frames(video_path: str, window_step: int) -> Generator[NoseTester, None, None]:
    capture = cv2.VideoCapture(video_path)

    # count the number of frames
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames == 0:
        capture.release()
        yield np.asarray([])

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

    x = math.ceil((len(resized_frames) - Config.window)/window_step)
    necessary = Config.window + x * window_step

    # copy the first frame and put this copies at the beginning
    for _ in range(necessary - len(resized_frames)):
        resized_frames.insert(0, np.copy(resized_frames[0]))

    # yield parts (windows)
    part = 0
    while True:
        if part * window_step + Config.window > len(resized_frames):
            break
        yield np.asarray(resized_frames[part * window_step : part * window_step + Config.window])
        part += 1


def create_data(paths: List[str], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_resized, y_onehot = [], []
    
    for index, vid_file in enumerate(paths):
        for part in get_frames(vid_file, Config.step):
            if part.size == 0:
                break
            assert part.shape == (Config.window, Config.img_dim, Config.img_dim, 3)
            x_resized.append(part)
            y_onehot.append(np.asarray(y[index], dtype=np.uint8))

    x_resized = np.asarray(x_resized)
    y_onehot = np.asarray(y_onehot)

    return x_resized, y_onehot


def save(x_train: np.ndarray, y_train: np.ndarray, x_test: List[str], y_test: np.ndarray) -> None:
    print('Saving...')
    if not os.path.exists(Config.data_path):
        os.makedirs(Config.data_path)
    
    np.save(os.path.join(Config.data_path, 'x_train.npy'), x_train)
    np.save(os.path.join(Config.data_path, 'y_train.npy'), y_train)
    np.save(os.path.join(Config.data_path, 'y_test.npy'), y_test)
    np.save(os.path.join(Config.data_path, 'classes.npy'), np.asarray(os.listdir(Config.dataset_path)))

    with open(os.path.join(Config.data_path, 'x_test.txt'), "w") as f:
        [f.write(path + '\n') for path in x_test]

    print('Done.')


# --------------- TRAINING ---------------

def load(data: List[str]) -> list:
    if not os.path.exists(Config.data_path):
        print('Your train/test data don\'t exist. First run \'python main.py prepare\' to prepare data before training.')
        sys.exit(0)

    matrices = {}

    for _, _, files in os.walk(Config.data_path):
        for file in files:
            if file.endswith(".npy") and file.split('.')[0] in data:
                matrices[file.split('.')[0]] = np.load(os.path.join(Config.data_path, file), allow_pickle=True)

    return [matrices[x] for x in data]


def create_model(class_num: int) -> Sequential:
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


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    print(x_train.shape)
    model = create_model(class_num=len(y_train[0]))
    opt = SGD(lr=3e-5, decay=3e-6)
    # opt = Adam(learning_rate=3e-6, lr=3e-6, decay=3e-7)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    if not os.path.exists(Config.model_path):
        os.makedirs(Config.model_path)

    checkpoints_path = os.path.join(Config.model_path, 'cp')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    cp_callback = ModelCheckpoint(filepath=checkpoints_path, save_weights_only=True, verbose=1)

    earlystop = EarlyStopping(patience=7)
    callbacks = [cp_callback, earlystop]

    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, train_size=Config.train_part, random_state=42)
    history = model.fit(x=x_fit, y=y_fit, epochs=40, batch_size=32, shuffle=True, callbacks=callbacks, validation_data=(x_val, y_val))

    model.save(Config.model_path + 'model.h5')

    try:
        model.save_weights(Config.model_path + 'weights')
        model.save(Config.model_path + 'model')
    except:
        pass

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(acc) + 1)

    # plt.plot(epochs, acc, 'r', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.figure()
    # plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.plot()
    # plt.legend()
    # plt.show()

    return model


# --------------- PREDICTION & STATISTICS ---------------

def evaluate_model(video_paths: List[str], y_test: np.ndarray) -> Tuple[float, float]:
    x_test, y_test = [], load(['y_test'])
    model = load_model(Config.model_path + 'model.h5')
    
    for index, vid_file in enumerate(video_paths):
        for part in get_frames(vid_file, Config.window):
            if part.size == 0:
                continue

            assert part.shape == (Config.window, Config.img_dim, Config.img_dim, 3)
            x_test.append(part)
            y_test.append(np.asarray(y_test[index], dtype=np.uint8))

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    loss, accuracy = model.evaluate(x_test)

    print(f'Loss={loss}, Accuracy={accuracy}')

    return loss, accuracy


def get_clsf_report(test_paths: List[str], y: np.ndarray) -> None:
    x_test, y_test = [], []
    model = load_model(Config.model_path + 'model.h5')
    
    for index, vid_file in enumerate(test_paths):
        for part in get_frames(vid_file, Config.window):
            if part.size == 0:
                continue
            
            assert part.shape == (Config.window, Config.img_dim, Config.img_dim, 3)
            x_test.append(part)
            y_test.append(np.asarray(y[index], dtype=np.uint8))

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    y_pred = model.predict(x_test)

    y_pred = np.argmax(y_pred, axis = 1)
    y_test = np.argmax(y_test, axis = 1)

    print(classification_report(y_test, y_pred))
    

def predict_class(video_path: str) -> str:
    model = load_model(Config.model_path + 'model.h5')

    if not os.path.exists(video_path):
        print(f'There is no video in {video_path}')
        sys.exit(0)

    classes = np.load(os.path.join(Config.data_path, 'classes.npy'), allow_pickle=True).tolist()
    
    y_preds = []
    for part in get_frames(video_path, Config.window):
        if part.size == 0:
            return
        y_pred = model.predict(np.asarray([part]))[0]
        y_preds.append(np.argmax(y_pred, axis = 1))
    
    index = max(set(y_preds), key=y_preds.count)
    predicted_class = classes[index]
    
    print('Predicted class =', predicted_class)

    return predicted_class


# -----------------------------------------------------------------------------------------------------------------------------

@typer_app.command('prepare')
def prepare() -> None:
    x_train_paths, x_test_paths, y_train, y_test = split()

    x_train, y_train = create_data(x_train_paths, y_train)
    save(x_train, y_train, x_test_paths, y_test)


@typer_app.command('train')
def train() -> None:
    x_train, y_train = load(['x_train', 'y_train'])
    
    train_model(x_train, y_train)


@typer_app.command('evaluate')
def evaluate() -> None:
    y_test = load(['y_test'])

    if not os.path.exists(os.path.join(Config.data_path, 'x_test.txt')):
        print(f'Missing {os.path.abspath(os.path.join(Config.data_path, "x_test.txt"))} containing test video paths.')
        sys.exit(0)

    with open(os.path.join(Config.data_path, 'x_test.txt'), "r") as f:
        content = f.readlines()

    evaluate_model(content[:10], y_test[:10])


@typer_app.command('predict')
def predict(src: str = typer.Option(None, '--path', '-p', help='Video path (REQUIRED).', show_default=False)) -> None:
    predict_class(src)

# -----------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    typer_app()
