from enum import Enum
from dataclasses import dataclass
from typing import Optional
import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

class Roles(Enum):
  REGULAR = 1
  VENDER = 2
  FREAK = 3

@dataclass
class Person:
  pid: int
  role: Roles = Roles.REGULAR
  household_id: Optional[int] = None


@dataclass
class Transaction:
  src: int
  dst: int
  timestamp: datetime
  amount: float


def date_range(start: date, end: date):
  for n in range((end - start).days + 1):
    yield start + timedelta(days=n)


def lognorm_amount(min_rub: float = 200.0, mean_rub: float = 10000.0, sigma: float = 0.7) -> float:
  mu = np.log(max(mean_rub, min_rub)) - (sigma**2)/2
  val = np.random.lognormal(mean=mu, sigma=sigma)
  return float(max(val, min_rub))


def random_dt_on(d: date, hour_mean=18, hour_std=3) -> datetime:
  h = int(np.clip(np.random.normal(hour_mean, hour_std), 8, 22))
  m = int(np.random.uniform(0, 60))
  s = int(np.random.uniform(0, 60))
  return datetime(d.year, d.month, d.day, h, m, s)


def populate(
    population: int = 100, 
    vender_frac: float = 0.01,
    freak_frac: float = 0.01,
    ) -> tuple[list[Person], dict[int, list[int]]]:
  
  freaks_num = int(population * freak_frac)
  ordinary_num = population 

  vender_ids = set(
    random.sample(
      range(ordinary_num), 
      int(ordinary_num * vender_frac)
    )
  )
  
  people: list[Person] = []
  for i in range(ordinary_num):
    people.append(Person(
      pid=i,
      role=Roles.VENDER if i in vender_ids else Roles.REGULAR
    ))

  households: dict[int, list[int]] = {}
  hid, remaining = 0, set(range(ordinary_num))
  while remaining:
    size = int(np.clip(np.random.normal(3, 1), 1, 5))
    members = random.sample(list(remaining), k=min(size, len(remaining)))
    households[hid] = members
    for pid in members:
      people[pid].household_id = hid
    remaining -= set(members)
    hid += 1

  return people, households


def simulate(
  people: list[Person], 
  households: dict[int, list[int]], 
  start_date: date, 
  end_date: date,
  hh_intensity: float = 0.6,      
  vender_workday_rate: float = 0.06, 
  vender_weekend_rate: float = 0.09,  
  noise_rate: float = 0.015,
) -> list[Transaction]:

  rows = []

  venders = [p for p in people if p.role == Roles.VENDER]

  for d in date_range(start_date, end_date):
    for _, members in households.items():
      if len(members) < 2:
        continue
      ops = np.random.poisson(hh_intensity)
      for _ in range(ops):
        src, dst = random.sample(members, 2)
        amount = lognorm_amount(
          min_rub=100.0, 
          mean_rub=15000.0, 
          sigma=0.7
        )
        rows.append(
          Transaction(
            src=src,
            dst=dst,
            amount=amount,
            timestamp=random_dt_on(d)
          )
        )
      
    if venders:
      # Считаем, что в выходные активность отличается
      if d.weekday() >= 5:
        ops = np.random.poisson(vender_weekend_rate*len(people))
      else:
        ops = np.random.poisson(vender_workday_rate*len(people))
      for _ in range(ops):
        src = random.choice(people)
        dst = random.choice(venders)
        if src.pid == dst.pid:
          continue
        amount = lognorm_amount(
          min_rub=1000.0, 
          mean_rub=5000.0, 
          sigma=0.7
        )
        rows.append(
          Transaction(
            src=src.pid,
            dst=dst.pid,
            amount=amount,
            timestamp=random_dt_on(d)
          )
        )
      
    # Сравнительно небольшой p2p-шум
    ops = np.random.poisson(noise_rate*len(people))
    for _ in range(ops):
      src, dst = random.sample(people, 2)
      if src.pid == dst.pid:
        continue
      amount = lognorm_amount(
        min_rub=100.0, 
        mean_rub=20000.0, 
        sigma=0.9
      )
      rows.append(
        Transaction(
          src=src.pid,
          dst=dst.pid,
          amount=amount,
          timestamp=random_dt_on(d)
        )
      )
  return rows


def make_undirectional(trxs: list[Transaction]) -> pd.DataFrame:
  trx_dict = {}
  for trx in trxs:
    key = (trx.src, trx.dst) if trx.src <= trx.dst else (trx.dst, trx.src)  
    if key in trx_dict:
      trx_dict[key].amount += trx.amount
    else:
      trx_dict[key] = trx
    
  return pd.DataFrame(trx_dict.values()).sort_values("timestamp").reset_index(drop=True)


def main():
  start_date = datetime(2025, 1, 1)
  end_date = datetime(2025, 1, 31)

  p, h = populate(
    population=100,
    vender_frac=0.05  
  )

  trxs = simulate(
    people=p,
    households=h,
    start_date=start_date,
    end_date=end_date,
    noise_rate=0.005
  )

  df = make_undirectional(trxs)
  df.to_csv('data/transactions.csv', index=False)  
  

if __name__ == '__main__':
  main()