import sys
import random

import pandas as pd

from phe import paillier

from utils import (
  load_pk,
  load_scores,
  how_long
)


def load_graph(path: str) -> pd.DataFrame:
  return pd.read_csv(path)


class PaymentService:
  def __init__(self, pk_path, graph_path: str, scores_path: str) -> None:
    self.pk = load_pk(pk_path)
    self.graph = load_graph(graph_path)
    self.scores = load_scores(self.pk, scores_path)
    self.masks = [random.randint(-sys.maxsize, sys.maxsize) for _ in range(len(self.scores))]
    self._compute()


  def _get_first_circle(self, pid: int) -> list[int]:
    df = self.graph.query(f'src == {pid} or dst == {pid}')
    return list(set(df[['src', 'dst']].values.flat) - set([pid]))
  

  def _compute(self):
    self.env_scores = {}
    # Цикл по всем людям
    for pid in self.scores.keys():
      friends = self._get_first_circle(pid)
      # Цикл по "друзьям"
      if len(friends) > 0:
        env_score = self.scores[friends[0]]
        for friend in friends[1:]:
          env_score += self.scores[friend]
        self.env_scores[pid] = env_score
  

  def simple_query(self, pid: int):
    return self.env_scores[pid]
  

  @how_long(resolution='seconds')
  def pir_ahe_query(self, idx: list[paillier.EncryptedNumber]) -> paillier.EncryptedNumber:
    r = idx[0] * self.env_scores[0].ciphertext()
    for i in range(1, len(idx)):
      r += idx[i]*self.env_scores[i].ciphertext()
    return r
  

  @how_long(resolution='seconds')
  def pir_query(self, idx: list[paillier.EncryptedNumber]) -> paillier.EncryptedNumber:
    r = idx[0] * self.masks[0]
    for i in range(1, len(idx)):
      r += idx[i] * self.masks[i]
    return r


  def get_masked_env_scores(self) -> list[paillier.EncryptedNumber]:
    return [self.env_scores[i] + self.masks[i] for i in range(len(self.scores))] 

  
if __name__ == '__main__':
  ps = PaymentService(
    pk_path='keys/pk.json',
    graph_path='data/transactions.csv',
    scores_path='data/scores.csv'
  )

  print(ps.simple_query(58))