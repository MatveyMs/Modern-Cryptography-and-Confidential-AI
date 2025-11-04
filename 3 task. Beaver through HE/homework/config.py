import os


PAILLIER_KEY_SIZE = 2048
MPC_MODULO = pow(2,64)
NUM_TRIPLES = int(os.getenv("NUM_TRIPLES", 100))