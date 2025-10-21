import os
import argparse
import pickle
import torch
import torch.distributed as dist

MOD_BITS = 32
MOD = 1 << MOD_BITS

def init_dist():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '2'))
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    return rank, world_size

def load_my_shares(rank):
    path = f'./triples/shares_party{rank}.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return int(data['a_share']), int(data['b_share'])

def load_my_triples(rank):
    path = f'./triples/beaver_party{rank}.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def bit_decompose(x):
    return [(x >> i) & 1 for i in range(MOD_BITS)]

def and_beaver(x_share, y_share, triple_local, rank):
    a_i, b_i, c_i = triple_local

    e_i = (x_share - a_i) % 2
    f_i = (y_share - b_i) % 2

    t_e = torch.tensor([e_i], dtype=torch.long)
    t_f = torch.tensor([f_i], dtype=torch.long)
    # раскрываем e,f
    dist.all_reduce(t_e, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_f, op=dist.ReduceOp.SUM)
    e = int(t_e.item() % 2)
    f = int(t_f.item() % 2)

    term_local = (c_i + (e * b_i) + (f * a_i)) % 2
    if rank == 0:
        term_local = (term_local + (e * f) % 2) % 2
    return int(term_local)

def not_share(x_share, rank):
    if rank == 0:
        return (1 - x_share) % 2
    else:
        return (-x_share) % 2

def xor_share(x_share, y_share):
    return (x_share + y_share) % 2

def compute_a_ge_b_bitwise(a_bits_share, b_bits_share, triples_local, rank):
    triple_idx = 0
    borrow_share = 0
    for i in range(MOD_BITS):
        a_i = a_bits_share[i]
        b_i = b_bits_share[i]

        not_a_i = not_share(a_i, rank)
        xor_i = xor_share(a_i, b_i)

        # t1 = (not a_i) AND b_i
        triple_t1 = triples_local[triple_idx]; triple_idx += 1
        t1_share = and_beaver(not_a_i, b_i, triple_t1, rank)

        # t2 = not(xor_i) AND borrow_share
        not_xor_i = not_share(xor_i, rank)
        triple_t2 = triples_local[triple_idx]; triple_idx += 1
        t2_share = and_beaver(not_xor_i, borrow_share, triple_t2, rank)

        # borrow_next = t1 OR t2 = t1 + t2 + t1*t2
        triple_t3 = triples_local[triple_idx]; triple_idx += 1
        borrow_and_share = and_beaver(t1_share, t2_share, triple_t3, rank)

        borrow_share = (t1_share + t2_share + borrow_and_share) % 2

    t = torch.tensor([borrow_share], dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    borrow = int(t.item() % 2)
    a_ge_b = (borrow == 0)
    return a_ge_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=False)
    args = parser.parse_args()

    rank, world_size = init_dist()
    print(f'Party {rank} started (rank {rank} / world {world_size})')

    a_share_int, b_share_int = load_my_shares(rank)
    triples_local = load_my_triples(rank)
    print(f'Party {rank}: loaded shares and {len(triples_local)} local Beaver triples')

    a_bits = bit_decompose(a_share_int)
    b_bits = bit_decompose(b_share_int)

    result = compute_a_ge_b_bitwise(a_bits, b_bits, triples_local, rank)

    if rank == 0:
        print('Result: a >= b ?', result)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()