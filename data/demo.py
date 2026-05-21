import os
import sys
import numpy as np


total_data = []

with open('dev.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('\n').strip()
        total_data.append(line)


res = np.array(total_data)
np.random.shuffle(res)

with open('dev1.txt', 'w', encoding='utf-8') as f1:
    for data in res:
        f1.write(data + '\n')


