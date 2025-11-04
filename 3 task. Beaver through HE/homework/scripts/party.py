import os
import csv
import argparse
import torch
import torch.distributed as dist
from phe import paillier
import secrets
import pickle
import config

def random_mod(mod):
    bits = mod.bit_length()
    while True:
        r = secrets.randbits(bits)
        if r < mod:
            return r

def init_process(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def send_pickle(obj, dst):
    data = pickle.dumps(obj)
    tensor = torch.tensor(list(data), dtype=torch.uint8)
    length_tensor = torch.tensor([len(tensor)], dtype=torch.int64)
    dist.send(length_tensor, dst=dst)
    dist.send(tensor, dst=dst)

def recv_pickle(src):
    length_tensor = torch.zeros(1, dtype=torch.int64)
    dist.recv(length_tensor, src=src)
    tensor = torch.zeros(length_tensor.item(), dtype=torch.uint8)
    dist.recv(tensor, src=src)
    data = bytes(tensor.tolist())
    obj = pickle.loads(data)
    return obj

def generate_beaver_triples(num_triples: int, modulo: int, rank: int, key_size: int = 2048):
    a_list: list[int] = []
    b_list: list[int] = []
    c_list: list[int] = []

    if rank == 0:
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)
        send_pickle(public_key, dst=1)
    else:
        public_key = recv_pickle(src=0)
        private_key = None

    for _ in range(num_triples):
        if rank == 0:
            a0 = random_mod(modulo)
            b0 = random_mod(modulo)

            enc_a0 = public_key.encrypt(a0)
            enc_b0 = public_key.encrypt(b0)
            send_pickle(enc_a0, dst=1)
            send_pickle(enc_b0, dst=1)

            enc_z = recv_pickle(src=1)
            z = private_key.decrypt(enc_z) % modulo   # z = a0*b1 + a1*b0 + a1*b1 + r (модуль q)

            # Добавляем свой член a0*b0 чтобы получить полное (a0+a1)*(b0+b1)
            z_total = (z + (a0 * b0)) % modulo

            c0 = random_mod(modulo)
            c1_to_send = (z_total - c0) % modulo
            send_pickle(c1_to_send, dst=1)

            a_list.append(a0)
            b_list.append(b0)
            c_list.append(c0)

        else:
            a1 = random_mod(modulo)
            b1 = random_mod(modulo)
            r  = random_mod(modulo)   # маска, чтобы P0 при дешифровке не увидел незамаскированные произведения

            enc_a0 = recv_pickle(src=0)
            enc_b0 = recv_pickle(src=0)

            # Гомоморфно вычисляем Enc(a0*b1 + b0*a1 + a1*b1 + r)
            # Операция __mul__ - перегружена в phe и даёт возведение в степень шифротекста
            # - enc_a0 * b1  => Enc(a0 * b1)
            # - enc_b0 * a1  => Enc(b0 * a1)
            enc_a0_b1 = enc_a0 * b1
            enc_b0_a1 = enc_b0 * a1
            enc_a1_b1 = public_key.encrypt((a1 * b1) % modulo)
            enc_r     = public_key.encrypt(r)

            enc_z = enc_a0_b1 + enc_b0_a1 + enc_a1_b1 + enc_r

            send_pickle(enc_z, dst=0)
            c1_from_p0 = recv_pickle(src=0)

            # компенсируем маску r — локально держим c1 = c1_from_p0 - r
            c1_local = (c1_from_p0 - r) % modulo

            a_list.append(a1)
            b_list.append(b1)
            c_list.append(c1_local)

    return a_list, b_list, c_list


def save_csv(a_list, b_list, c_list, rank):
    os.makedirs("triples", exist_ok=True)
    filename = f"triples/p{rank}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for a, b, c in zip(a_list, b_list, c_list):
            writer.writerow([a, b, c])
    print(f"[rank{rank}] Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()

    rank = args.rank
    world_size = int(os.getenv("WORLD_SIZE", 2))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "12355")
    num_triples = int(os.getenv("NUM_TRIPLES", 100))

    init_process(rank, world_size, master_addr, master_port)
    a_list, b_list, c_list = generate_beaver_triples(num_triples, config.MPC_MODULO, rank, key_size=config.PAILLIER_KEY_SIZE)
    save_csv(a_list, b_list, c_list, rank)
    dist.destroy_process_group()