class Config:
    train_part = 0.8
    fps_to_take = 5
    step = 4
    window = 6
    img_dim = 64
    dataset_path = '../data/dataset'
    data_path = f'../data/train_test_{img_dim}x{img_dim}'
    model_path = f'../data/model_{img_dim}x{img_dim}/'