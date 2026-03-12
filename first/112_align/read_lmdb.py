import cv2
import lmdb
import numpy as np
import os
import matplotlib.pyplot as plt

root = r'/home/qitam/sdb2/home/qiteam_project/lhc/project/ABAW3/submission/LMDB'
env = lmdb.open(os.path.join(root, '.label_au'))
env_va = lmdb.open(os.path.join(root, '.label_va'))
env_expr = lmdb.open(os.path.join(root, '.label_expr'))

txn = env.begin()

count = 0
# values = np.zeros(12)
# values_n = np.zeros(12)
# values = np.zeros(8)
values = 0
values_a = 0
values_a_v = 0
values_a_v_e = 0
values_a_e = 0
values_v = 0
values_v_e = 0
values_e = 0
for key, value in env_expr.begin().cursor():  # 遍历
    # value = np.frombuffer(value, dtype=np.float32)
    l = np.frombuffer(value, dtype=np.int8)

    if l == -1:
        continue
    count += 1
    # values += 1
    # if value_int == 255:
    #     value_int = 7
    # values[value_int == 1] += 1
    # values_n[value_int == 0] += 1
    # if value[0] == -5:
    #     continue
    #

    with env.begin(write=False) as txn:
        l_a = txn.get(key)
        if l_a == None:
            continue
        l_a = np.frombuffer(l_a, dtype=np.int8)
        if l_a[0] == -1:
            continue
    with env_va.begin(write=False) as txn_v:
        l_v = txn_v.get(key)
        if l_v == None:
            continue
        l_v = np.frombuffer(l_v, dtype=np.float32)
        if l_v[0] == -5:
            continue
    # with env_expr.begin(write=False) as txn_e:
    #     l_e = txn_e.get(key)
    #     if l_e == None:
    #         continue
    #     l_e = np.frombuffer(l_e, dtype=np.int8).item()
    #     if l_e == -1:
    #         continue
    values += 1
    # if l_a[0] == -1:
    #     if l_v[0] == -5:
    #         if l_e == -1:
    #             values += 1
    #         else:
    #             values_e += 1
    #     else:
    #         if l_e == -1:
    #             values_v += 1
    #         else:
    #             values_v_e += 1
    # else:
    #     if l_v[0] == -5:
    #         if l_e == -1:
    #             values_a += 1
    #         else:
    #             values_a_e += 1
    #     else:
    #         if l_e == -1:
    #             values_a_v += 1
    #         else:
    #             values_a_v_e += 1

# plt.figure()
# plt.bar(range(len(values)), values)
# plt.bar(range(len(values_n)), values_n, bottom=values)
# plt.show()
print(count)

print(values)
print(values_a)
print(values_a_v)
print(values_a_v_e)
print(values_a_e)
print(values_v)
print(values_v_e)
print(values_e)