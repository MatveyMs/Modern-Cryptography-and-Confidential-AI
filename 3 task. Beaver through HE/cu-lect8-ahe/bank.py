from phe import paillier

from utils import (
  load_pk, 
  load_sk, 
  load_scores,
  how_long
)

class Bank:
  def __init__(
      self, 
      pk_path: str, 
      sk_path: str, 
      scores_path: str):
    self.pk = load_pk(pk_path)
    self.sk = load_sk(sk_path, self.pk,)
    self.pkx, self.skx = paillier.generate_paillier_keypair(n_length=4112)
    self.scores = load_scores(self.pk, scores_path)

  
  def decrypt(self, v: paillier.EncryptedNumber) -> int:
    return self.sk.decrypt(v)
  

  @how_long(resolution='seconds')
  def make_pir_query(self, pid: int) -> list[paillier.EncryptedNumber]:
    vec = []
    for i in range(len(self.scores)):
      vec.append(self.pk.encrypt(1 if i == pid else 0))
    return vec
  

  def decrypt_pir(self, masked_score: paillier.EncryptedNumber, mask: paillier.EncryptedNumber) -> int:
    masked_score = self.sk.decrypt(masked_score)
    mask = self.sk.decrypt(mask)
    return masked_score - mask
  

  @how_long(resolution='seconds')
  def make_pir_query_ahe(self, pid: int) -> list[paillier.EncryptedNumber]:
    vec = []
    for i in range(len(self.scores)):
      vec.append(self.pkx.encrypt(1 if i == pid else 0))
    return vec


  def decrypt_ahe(self, v: paillier.EncryptedNumber) -> int:
    return self.sk.decrypt(
      paillier.EncryptedNumber(
        public_key=self.pk,
        ciphertext=self.skx.decrypt(v)
      )
    )


def main():
  b = Bank(
    pk_path='keys/pk.json', 
    sk_path='keys/sk.json', 
    scores_path='data/scores.csv'
  ) 


if __name__ == '__main__':
  main()