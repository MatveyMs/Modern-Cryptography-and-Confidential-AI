import time
import logging
import torch
import crypten
import crypten.communicator as comm
import crypten.nn as nn
from crypten.nn.loss import MSELoss
from crypten.optim import SGD
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

from config import config
from . import task

def init():
    crypten.init()
    comm.get().set_verbosity(True)
    crypten.encoder.set_default_precision(config.PRECISION)

@task("ttp")
def ttp():
    crypten.mpc.provider.TTPServer()

@task("mul")
def mul():
    init()
    rank = comm.get().get_rank()

    tensor_size = 4
    if rank == 0:
        x = torch.arange(tensor_size) * 1_000_000
        y = torch.zeros(0)
    elif rank == 1:
        x = torch.zeros(0)
        y = torch.arange(tensor_size) + 1

    x_enc = ArithmeticSharedTensor(x, src=0, broadcast_size=True)
    y_enc = ArithmeticSharedTensor(y, src=1, broadcast_size=True)

    a_enc = BinarySharedTensor(x, src=0, broadcast_size=True)
    b_enc = BinarySharedTensor(y, src=1, broadcast_size=True)

    logging.info(f"Rank {rank}: x share = {x_enc.share}")
    logging.info(f"Rank {rank}: y share = {y_enc.share}")

    comm.get().reset_communication_stats()
    z_enc = x_enc * y_enc
    comm.get().print_communication_stats()

    comm.get().reset_communication_stats()
    c_enc = a_enc > b_enc
    comm.get().print_communication_stats()

    logging.info(f"Rank {rank}: z share = {z_enc.share}")
    logging.info(f"Rank {rank}: c share = {c_enc.share}")

    z = z_enc.get_plain_text()
    c = c_enc.get_plain_text()
    logging.info(f"Rank {rank}: z plaintext = {z}")
    logging.info(f"Rank {rank}: c plaintext = {c}")

    crypten.uninit()

@task("linreg")
def linreg():
    init()

    # Создаём mpc-тензоры
    X_train_enc = crypten.load_from_party(config.X_TRAIN_PATH, src=0)
    X_test_enc  = crypten.load_from_party(config.X_TEST_PATH, src=0)
    y_train_enc = crypten.load_from_party(config.Y_TRAIN_PATH, src=1)
    y_test_enc  = crypten.load_from_party(config.Y_TEST_PATH, src=1)

    # Модель
    class LinearModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.fc(x)

    input_dim = X_train_enc.shape[1]
    model = LinearModel(input_dim)

    # инициализируем веса
    for name, weight in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(weight, mean=0.0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(weight, 0.)
    model.encrypt()  # переводим веса модели в разделения секрета

    # Обучение
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)

    batch_size = config.BATCH_SIZE
    n_epochs = config.EPOCHS

    n_samples = y_train_enc.size(0)
    n_batches = (n_samples + batch_size - 1) // batch_size

    t0 = time.time()

    comm.get().reset_communication_stats()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i in range(n_batches):
            start, end = i * batch_size, min((i+1) * batch_size, n_samples)

            X_batch = X_train_enc[start:end]
            y_batch = y_train_enc[start:end]

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.get_plain_text().item()

        if (epoch+1) % 5 == 0 or epoch == 0:
            logging.info(f"[Epoch {epoch+1:02d}] loss={epoch_loss/n_batches:.6f}")

    logging.info(f"Model train elapsed time: {time.time() - t0:.4f} seconds")
    comm.get().print_communication_stats()
    # Валидация на X_test
    with torch.no_grad():
        y_pred_enc = model(X_test_enc)
        test_loss = criterion(y_pred_enc, y_test_enc)
        logging.info(f"Test MSE: {test_loss.get_plain_text().item()}")

    # Веса модели
    w_learned_enc = model.fc.weight
    b_learned_enc = model.fc.bias

    logging.info(f"Learned weights share: {w_learned_enc.share.view(-1)}")
    logging.info(f"Learned bias share: {b_learned_enc.share.item()}")
    
    crypten.uninit()


@task("linreg_aug")
def linreg_aug():
    init()
    rank = comm.get().get_rank()

    # Пути к тренировочным и тестовым файлам внутри контейнера
    path_x1 = "/data/linreg/x_train_norm_worker1.pth"
    path_y1 = "/data/linreg/y_train_worker1.pth"
    path_x2 = "/data/linreg/x_train_norm_worker2.pth"
    path_y2 = "/data/linreg/y_train_worker2.pth"
    path_x_test = "/data/linreg/x_test_norm_worker1.pth"
    path_y_test = "/data/linreg/y_test_worker1.pth"


    X1_enc = crypten.load_from_party(path_x1, src=0)  # data owned by party 0
    y1_enc = crypten.load_from_party(path_y1, src=0)
    X2_enc = crypten.load_from_party(path_x2, src=1)  # data owned by party 1
    y2_enc = crypten.load_from_party(path_y2, src=1)

    X_test_enc = crypten.load_from_party(path_x_test, src=0)
    y_test_enc = crypten.load_from_party(path_y_test, src=0)

    class LinearModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.fc(x)

    input_dim = X1_enc.shape[1] if X1_enc.shape[1] is not None else X2_enc.shape[1]
    model = LinearModel(input_dim)

    for name, weight in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(weight, mean=0.0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(weight, 0.0)
    model.encrypt()

    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)

    n1 = y1_enc.size(0)
    n2 = y2_enc.size(0)
    n_total = n1 + n2

    n_epochs = config.EPOCHS

    t0 = time.time()
    comm.get().reset_communication_stats()

    for epoch in range(n_epochs):
        preds1 = model(X1_enc)
        loss1 = criterion(preds1, y1_enc)

        preds2 = model(X2_enc)
        loss2 = criterion(preds2, y2_enc)

        total_loss = (loss1 * float(n1) + loss2 * float(n2)) / float(n_total)

        model.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            try:
                loss_val = total_loss.get_plain_text().item()
            except Exception:
                loss_val = None
            logging.info(f"[Epoch {epoch+1:02d}] avg_loss={loss_val}")

    logging.info(f"Model train elapsed time: {time.time() - t0:.4f} seconds")
    comm.get().print_communication_stats()

    with torch.no_grad():
        y_pred_enc = model(X_test_enc)
        test_loss = criterion(y_pred_enc, y_test_enc)
        logging.info(f"Test MSE (encrypted): {test_loss.get_plain_text().item()}")

    w_learned_enc = model.fc.weight
    b_learned_enc = model.fc.bias

    w_plain = model.fc.weight.get_plain_text()
    b_plain = model.fc.bias.get_plain_text()

    if rank == 0:
        time.sleep(5)
        logging.info(f"Learned weights (plaintext): {w_plain.view(-1)}")
        logging.info(f"Learned bias (plaintext): {b_plain.item()}")
    elif rank == 1:
        time.sleep(10)
    logging.info(f"Learned weights share: {w_learned_enc.share.view(-1)}")
    logging.info(f"Learned bias share: {b_learned_enc.share.item()}")


    crypten.uninit()