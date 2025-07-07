import os
import pickle
import numpy as np
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

CIF_DIR = "data/testcif/"
OUTPUT_DIR = "data/"
DATASET_FILE = "data/my_custom_data.p"

def load_and_prepare_data():
    """加载CIF文件并创建数据集"""
    dataset = {}
    
    print("开始加载CIF文件...")
    for cif_file in os.listdir(CIF_DIR):
        if not cif_file.endswith(".cif"):
            continue
            
        try:
            # 从CIF文件加载结构
            struct = Structure.from_file(os.path.join(CIF_DIR, cif_file))
            material_id = os.path.splitext(cif_file)[0]
            
            # 添加到数据集
            dataset[material_id] = {
                "structure": struct,
                # 这些值需要替换为实际数据
                "energy": 0.0,        
                "force": np.zeros((len(struct), 3))
            }
            
        except Exception as e:
            print(f"处理文件 {cif_file} 时出错: {str(e)}")
    
    print(f"成功加载 {len(dataset)} 个材料")
    return dataset

def split_dataset(dataset):
    """将数据集分割为训练集、验证集和测试集"""
    material_ids = list(dataset.keys())
    
    # 第一次分割：训练集 (80%) 和 临时集 (20%)
    train_ids, temp_ids = train_test_split(material_ids, test_size=0.2, random_state=42)
    
    # 第二次分割：验证集 (10%) 和测试集 (10%)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    # 创建分割后的数据集
    train_data = {mid: dataset[mid] for mid in train_ids}
    val_data = {mid: dataset[mid] for mid in val_ids}
    test_data = {mid: dataset[mid] for mid in test_ids}
    
    print(f"数据集分割结果: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")
    return train_data, val_data, test_data

def save_datasets(train_data, val_data, test_data):
    """保存分割后的数据集"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, "train_set.p"), "wb") as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(OUTPUT_DIR, "val_set.p"), "wb") as f:
        pickle.dump(val_data, f)
    
    with open(os.path.join(OUTPUT_DIR, "test_set.p"), "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"数据集已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    # 1. 加载和准备数据
    dataset = load_and_prepare_data()
    
    # 2. 分割数据集
    train_data, val_data, test_data = split_dataset(dataset)
    
    # 3. 保存分割后的数据集
    save_datasets(train_data, val_data, test_data)
    
    # 4. 保存完整数据集（可选）
    with open(DATASET_FILE, "wb") as f:
        pickle.dump(dataset, f)
    print(f"完整数据集已保存至 {DATASET_FILE}")
