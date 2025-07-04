import pickle as pk
import pandas as pd
import numpy as np
import tensorflow as tf
from m3gnet.models import M3GNet, Potential
from m3gnet.trainers import PotentialTrainer
import pymatgen

print('加载MPF 2021数据集')
with open('data/block_0.p', 'rb') as f:
    data = pk.load(f)

with open('data/block_1.p', 'rb') as f:
    data2 = pk.load(f)
print('MPF 2021数据集加载完成')
data.update(data2)

def get_id_train_val_test(
    total_size: int,
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    keep_data_order: bool = False
):
    """
    分割数据集索引为训练/验证/测试集
    """
    assert train_ratio + val_ratio + test_ratio == 1
    
    indices = np.arange(total_size)
    if not keep_data_order:
        rng = np.random.RandomState(split_seed)
        rng.shuffle(indices)
    
    train_end = int(train_ratio * total_size)
    val_end = train_end + int(val_ratio * total_size)
    
    return (
        indices[:train_end].tolist(),
        indices[train_end:val_end].tolist(),
        indices[val_end:].tolist()
    )

id_train, id_val, id_test = get_id_train_val_test(
    total_size=len(data),
    split_seed=42,
    train_ratio=0.90,
    val_ratio=0.05,
    test_ratio=0.05,
    keep_data_order=False,
)

# 初始化数据集列表
dataset_train = []
dataset_val = []
dataset_test = []

cnt = 0
for key, item in data.items():
    if cnt in id_train:
        target_list = dataset_train
    elif cnt in id_val:
        target_list = dataset_val
    elif cnt in id_test:
        target_list = dataset_test
    
    # 处理每个结构的数据
    for iid in range(len(item['energy'])):
        target_list.append({
            "atoms": item['structure'][iid],
            "energy": item['energy'][iid] / len(item['force'][iid]),
            "force": np.array(item['force'][iid])
        })
    
    cnt += 1

print(f'使用 {len(dataset_train)} 个样本训练, {len(dataset_val)} 个样本验证, {len(dataset_test)} 个样本测试')

# 准备训练数据
def extract_data(dataset):
    structures = [d["atoms"] for d in dataset]
    energies = [d["energy"] for d in dataset]
    forces = [d["force"] for d in dataset]
    return structures, energies, forces

train_structures, train_energies, train_forces = extract_data(dataset_train)
val_structures, val_energies, val_forces = extract_data(dataset_val)

# 初始化模型和训练器
m3gnet = M3GNet(is_intensive=False)
potential = Potential(model=m3gnet)

trainer = PotentialTrainer(
    potential=potential,
    optimizer=tf.keras.optimizers.Adam(1e-3)
)

# 开始训练
trainer.train(
    train_structures,
    train_energies,
    train_forces,
    validation_graphs_or_structures=val_structures,
    val_energies=val_energies,
    val_forces=val_forces,
    epochs=100,
    fit_per_element_offset=True,
    save_checkpoint=True
)
