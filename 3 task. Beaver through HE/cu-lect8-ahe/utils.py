import time
import json
import csv
from datetime import timedelta
from functools import wraps
import humanize
from phe import paillier


def how_long(resolution='seconds'):
  def _decorator(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
      start_time = time.perf_counter()
      result = func(*args, **kwargs)
      end_time = time.perf_counter()
      total_time = end_time - start_time
      interval = humanize.precisedelta(
        timedelta(seconds=total_time),
        minimum_unit=resolution,
        format='%0.4f'
      )
      print(f'func "{func.__name__}" took {interval} to execute.')
      return result
    return _wrapper
  return _decorator


def load_pk(path: str) -> paillier.PaillierPublicKey:
  with open(path, 'r') as file:
    obj = json.load(file)
    pk = paillier.PaillierPublicKey(n=obj['n'])
    pk.g = obj['g']
  return pk


def load_sk(path: str, pk: paillier.PaillierPublicKey) -> paillier.PaillierPrivateKey:
  with open(path, 'r') as file:
    obj = json.load(file)
    sk = paillier.PaillierPrivateKey(pk, p=obj['p'], q=obj['q'])
  return sk


def load_scores(
    pk: paillier.PaillierPublicKey, 
    path: str
  ) -> dict[int,paillier.EncryptedNumber]:
  d = {}
  with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)
    for entry in reader:
      d[int(entry[0])] = paillier.EncryptedNumber(
        public_key=pk,
        ciphertext=int(entry[2])
      )
  return d