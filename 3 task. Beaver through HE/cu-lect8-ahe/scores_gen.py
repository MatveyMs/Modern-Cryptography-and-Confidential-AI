import json
import random
import decimal

import pandas as pd
from phe import paillier

from utils import how_long


@how_long(resolution='seconds')
def generate_keys() -> tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
  pk, sk = paillier.generate_paillier_keypair(n_length=2048)
  return pk, sk


@how_long(resolution='seconds')
def generate_scores(
  pk: paillier.PaillierPublicKey,
  population: int = 100
) -> pd.DataFrame:
  
  rows = []
  for i in range(population):
    score_plain = random.randint(1, 5)
    score_enc = pk.encrypt(score_plain)
    rows.append({
      'pid': i,
      'score_plain': score_plain,
      'score_enc': decimal.Decimal(score_enc.ciphertext())
    })
  return pd.DataFrame(rows).reset_index(drop=True)


def save_keys(
  pk_path: str,
  sk_path: str,
  pk: paillier.PaillierPublicKey,
  sk: paillier.PaillierPrivateKey
):
  with open(pk_path, 'w') as file:
    json.dump(
      obj = {
        'g': pk.g,
        'n': pk.n
      }, 
      fp=file
    )

  with open(sk_path, 'w') as file:
    json.dump(
      obj = {
        'p': sk.p,
        'q': sk.q
      },
      fp=file
    )


def main():
  print('generate keys...')

  pk, sk = generate_keys()

  save_keys(
    pk_path='keys/pk.json',
    sk_path='keys/sk.json',
    pk=pk, 
    sk=sk
  )

  print('encrypt scores...')

  df = generate_scores(pk)

  v = paillier.EncryptedNumber(pk, int(df['score_enc'].iloc[0]))
  print(sk.decrypt(v))

  df.to_csv('data/scores.csv', index=False)


if __name__ == '__main__':
    main()