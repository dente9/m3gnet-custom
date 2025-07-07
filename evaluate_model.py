import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from m3gnet.models import M3GNet
from m3gnet.trainers import PotentialTrainer
from m3gnet.trainers._potential import Potential  # 修正Potential导入路径
from m3gnet.graph import MaterialGraph, RadiusCutoffGraphConverter

def evaluate_model(model, test_data):
    """
    自定义模型评估函数
    """
    energy_errors = []
    force_errors = []
    converter = RadiusCutoffGraphConverter()  # 创建转换器
    
    for sample in test_data:
        # 使用转换器创建材料图
        graph = converter.convert(sample["atoms"])
        # 正确提取TensorFlow张量
        inputs = (
            graph.atoms,
            graph.bonds,
            graph.atom_positions,
            graph.bond_atom_indices,
            graph.pbc_offsets,
            graph.n_atoms,
            graph.n_bonds,
            graph.bond_weights,
            graph.lattices,
            graph.triple_bond_indices,
            graph.n_triple_ij,
            graph.n_triple_i,
            graph.n_triple_s
        )
        
        # 预测
        pred = model(inputs)
        
        # 计算误差
        energy_err = pred[0].numpy()[0][0] - sample["energy"]
        force_err = np.abs(pred[1].numpy()[0] - sample["force"]).flatten()
        
        energy_errors.append(energy_err)
        force_errors.extend(force_err)
    
    return {
        "energy_mae": np.mean(np.abs(energy_errors)),
        "energy_rmse": np.sqrt(np.mean(np.square(energy_errors))),
        "force_mae": np.mean(np.abs(force_errors)),
        "energy_errors": energy_errors,
        "force_errors": force_errors  # 添加力误差数据
    }

def find_latest_checkpoint(checkpoint_dir="callbacks"):
    """查找最新的模型检查点"""
    # 获取所有检查点文件
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".index")]
    if not ckpt_files:
        raise FileNotFoundError("在回调目录中找不到检查点文件")
    
    # 按训练步数排序
    ckpt_files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    # 返回完整的检查点前缀（不带扩展名）
    return os.path.join(checkpoint_dir, ckpt_files[-1].replace(".index", ""))

# 1. 加载模型
try:
    # 尝试加载最新检查点
    latest_ckpt = find_latest_checkpoint()
    print(f"加载最新检查点: {latest_ckpt}")
    
    # 创建模型实例 - 指定与训练时相同的配置
    model = M3GNet(
        n_atom_types=10,  # 原子类型数量为10
        is_intensive=False,
        n_blocks=3,
        units=64,
        cutoff=5.0
    )
    
    # 创建Potential包装器
    potential = Potential(model=model)
    
    # 加载权重
    potential.load_weights(latest_ckpt)
    model = potential.model  # 获取内部模型
    print("模型权重加载成功！")
except Exception as e:
    print(f"加载模型失败: {str(e)}")
    exit(1)

# 2. 加载测试数据
try:
    # 加载测试集 - 使用新生成的数据集
    with open('data/test_set.p', 'rb') as f:
        test_data_dict = pk.load(f)
    
    # 将字典转换为列表格式
    test_data = []
    for material_id, data in test_data_dict.items():
        test_data.append({
            "atoms": data["structure"],
            "energy": data["energy"],
            "force": data["force"]
        })
    print(f"成功加载测试集，包含 {len(test_data)} 个样本")
except Exception as e:
    print(f"加载测试集失败: {str(e)}")
    exit(1)

# 3. 计算性能指标
results = evaluate_model(model, test_data)

# 4. 输出报告
print("\n=== 模型评估报告 ===")
print(f"能量平均绝对误差 (MAE): {results['energy_mae']:.4f} eV")
print(f"力分量平均绝对误差 (MAE): {results['force_mae']:.4f} eV/Å")
print(f"能量均方根误差 (RMSE): {results['energy_rmse']:.4f} eV")
# 力分量均方根误差将在步骤7计算并输出

# 5. 可视化误差分布
plt.figure(figsize=(10, 6))
plt.hist(results['energy_errors'], bins=50, alpha=0.7, color='blue')
plt.xlabel('能量预测误差 (eV)')
plt.ylabel('样本数量')
plt.title('能量预测误差分布')
plt.grid(True, alpha=0.3)
plt.savefig('energy_error_distribution.png')
print("\n误差分布图已保存至 energy_error_distribution.png")

# 6. 保存完整结果
with open('evaluation_results.pkl', 'wb') as f:
    pk.dump(results, f)
print("完整评估结果已保存至 evaluation_results.pkl")

# 7. 添加力分量均方根误差计算
force_errors_arr = np.array(results.get('force_errors', []))
if len(force_errors_arr) > 0:
    results['force_rmse'] = np.sqrt(np.mean(np.square(force_errors_arr)))
    # 在报告末尾输出力分量RMSE
    print(f"力分量均方根误差 (RMSE): {results['force_rmse']:.4f} eV/Å")
else:
    print("无法计算力分量均方根误差 - 无有效数据")
