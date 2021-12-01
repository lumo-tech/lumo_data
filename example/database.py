from dbrecord import PDict
from lumo_data import DBDataset, DataLoader
import os

if os.path.exists('temp.sql'):
    os.remove('temp.sql')
dic = PDict('temp.sql')
for i in range(1000):
    dic[f'a{i}'] = i
dic.flush()
dataset = DBDataset('temp.sql')
for batch in DataLoader(dataset, num_workers=1, batch_size=16):
    pass
