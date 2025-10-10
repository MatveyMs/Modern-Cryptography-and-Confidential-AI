import os


class Config:
    # Логирование
    LOGLEVEL = os.getenv("LOGLEVEL", "INFO")
    # computation
    X_TRAIN_PATH = os.getenv("X_TRAIN_PATH", "/data/linreg/x_train.pth")
    X_TEST_PATH = os.getenv("X_TEST_PATH", "/data/linreg/x_test.pth")
    Y_TRAIN_PATH = os.getenv("Y_TRAIN_PATH", "/data/linreg/y_train.pth")
    Y_TEST_PATH = os.getenv("Y_TEST_PATH", "/data/linreg/y_test.pth")
    PRECISION = int(os.getenv("PRECISION", 16))
    EPOCHS = int(os.getenv("EPOCHS", 20))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 512))



config = Config()
