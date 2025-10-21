import os
import pickle
import numpy as np


MOD_BITS = 32
MOD = 1 << MOD_BITS
BEAVER_COUNT = 3 * MOD_BITS + 50

OUT_DIR = './triples'
os.makedirs(OUT_DIR, exist_ok=True)

def int_to_bits(x, bits=MOD_BITS):
    return [(x >> i) & 1 for i in range(bits)]

def bits_to_int(bits):
    v = 0
    for i, b in enumerate(bits):
        v |= (int(b) & 1) << i
    return v

def generate_bitwise_shares(value):
    bits = int_to_bits(value, MOD_BITS)
    bits0 = []
    bits1 = []
    for bit in bits:
        b0 = int(np.random.randint(0,2))
        b1 = (bit - b0) % 2
        bits0.append(b0)
        bits1.append(b1)
    return bits_to_int(bits0), bits_to_int(bits1)

def generate_beaver_triples(count):
    triples0 = []
    triples1 = []
    for _ in range(count):
        a = int(np.random.randint(0,2))
        b = int(np.random.randint(0,2))
        c = (a * b) % 2
        a0 = int(np.random.randint(0,2))
        b0 = int(np.random.randint(0,2))
        c0 = int(np.random.randint(0,2))
        a1 = (a - a0) % 2
        b1 = (b - b0) % 2
        c1 = (c - c0) % 2
        triples0.append((a0, b0, c0))
        triples1.append((a1, b1, c1))
    return triples0, triples1

def main():
    a_true = int(os.environ.get('A_TRUE', '99'))
    b_true = int(os.environ.get('B_TRUE', '100'))

    a_true = a_true % MOD
    b_true = b_true % MOD

    a0_int, a1_int = generate_bitwise_shares(a_true)
    b0_int, b1_int = generate_bitwise_shares(b_true)

    shares0 = {'a_share': int(a0_int), 'b_share': int(b0_int)}
    shares1 = {'a_share': int(a1_int), 'b_share': int(b1_int)}

    with open(os.path.join(OUT_DIR, 'shares_party0.pkl'), 'wb') as f:
        pickle.dump(shares0, f)
    with open(os.path.join(OUT_DIR, 'shares_party1.pkl'), 'wb') as f:
        pickle.dump(shares1, f)

    triples0, triples1 = generate_beaver_triples(BEAVER_COUNT)
    with open(os.path.join(OUT_DIR, 'beaver_party0.pkl'), 'wb') as f:
        pickle.dump(triples0, f)
    with open(os.path.join(OUT_DIR, 'beaver_party1.pkl'), 'wb') as f:
        pickle.dump(triples1, f)

    print(f"TTP: wrote shares and {len(triples0)} Beaver triples per party into {OUT_DIR}")
    print(f"  a_true={a_true}, b_true={b_true}")
    print(f"  a0={a0_int}, a1={a1_int}")
    print(f"  b0={b0_int}, b1={b1_int}")

if __name__ == '__main__':
    main()