
#################### 导入需要的python库 ####################
## run start ##
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import numpy as np
import pandas as pd

from tableone import TableOne
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import multivariate_logrank_test
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator
import shap
## run end ##

####################### 数据导入 #######################

# 导入数据
data = pd.read_csv("data/gbsg2_survival.csv", encoding="utf-8-sig", index_col=0)  # id作为索引，不参与建模
# 查看数据情况
print(data.head())  # 查看前五行数据
print(data.info())  # 查看数据结构

#####################################################################################
################################# 数据理解和探索 ######################################
#####################################################################################
#################### 查看结局变量与生存时间 ####################
## 统计结局变量cens的频数和频率（1=事件发生，0=右删失） ##
print(data['cens'].value_counts().sort_index())  # 频数
print(data['cens'].value_counts(normalize=True).sort_index())  # 频率

## 查看生存时间time的分布（单位：天） ##
print(data['time'].describe())

#################### 检查数据情况 ####################
# 检查数据集中缺失值的情况
missing_values = data.isnull().sum()
print(missing_values)  # 无缺失
# 检查患者编号、生存时间和结局状态
assert data.index.is_unique  # 每名患者对应一条记录
assert (data['time'] > 0).all()  # 生存时间必须为正数
assert data['cens'].isin([0, 1]).all()  # cens：1=事件发生，0=右删失

#####################################################################################
#################################### 数据拆分 ########################################
#####################################################################################
##################### 1. 数据集拆分：开发集和测试集 ######################
development_data, test_data = train_test_split(data, test_size=0.3,  # 拆分数据
                                               stratify=data["cens"], random_state=2025)

######################## 2. 查看开发集vs测试集的变量分布 ########################
# 为两个数据集都生成一个group变量，便于后续变量分布检查
development_table = development_data.copy()
test_table = test_data.copy()
development_table["group"] = "development_set"
test_table["group"] = "test_set"
# 合并这两个数据集
total = pd.concat([development_table, test_table])  # 默认按行合并
# 创建描述性统计表
categorical_vars = ["cens", "horTh", "menostat", "tgrade"]  # 分类变量的变量名
all_vars = total.columns.values[0:len(total.columns)-1].tolist()  # 除去'group'变量外的所有变量名
varbalance_table = TableOne(data=total, columns=all_vars,
                            categorical=categorical_vars, groupby="group", pval=False)  # 创建描述性统计表
# 查看变量分布情况
print(varbalance_table)
varbalance_table
# 保存为csv文件，方便查看和粘贴
os.makedirs("outputs/tables", exist_ok=True)  # 创建表格输出目录
varbalance_table.to_csv("outputs/tables/varbalance_table.csv", encoding="utf-8-sig")

##################### 3. 数据集拆分：训练集和验证集 ######################
train_data, val_data = train_test_split(development_data, test_size=0.3,  # 拆分数据
                                       stratify=development_data["cens"], random_state=123)

# 保存这三个数据集
os.makedirs("outputs/data_splits", exist_ok=True)  # 创建数据拆分输出目录
train_data.to_csv("outputs/data_splits/train_raw.csv")  # 保存训练集
val_data.to_csv("outputs/data_splits/validation_raw.csv")  # 保存验证集
test_data.to_csv("outputs/data_splits/test_raw.csv")  # 保存测试集

# 查看三个数据集的样本量和事件比例
print("Sample sizes:", len(train_data), len(val_data), len(test_data))  # 训练集、验证集、测试集样本量
print("Event rates:", train_data["cens"].mean(), val_data["cens"].mean(), test_data["cens"].mean())  # 事件发生比例

#####################################################################################
#################################### 特征工程 ########################################
#####################################################################################
################ 1. 连续型变量标准化，后续加快机器学习模型收敛 #################
## 首先训练集连续变量的标准化
train_data_scaled = train_data.copy()
val_data_scaled = val_data.copy()
test_data_scaled = test_data.copy()

continuous_vars = ['age', 'tsize', 'pnodes', 'progrec', 'estrec']  # 连续变量的变量名
scaler = StandardScaler()
train_data_scaled[continuous_vars] = scaler.fit_transform(train_data_scaled[continuous_vars])  # 对训练集中的连续变量进行标准化
train_data_scaled
# 保存标准化后的训练集
os.makedirs("outputs/data_processed", exist_ok=True)  # 创建数据输出目录
train_data_scaled.to_csv("outputs/data_processed/train_scaled.csv")

## 接下来验证集和测试集连续变量的标准化
val_data_scaled[continuous_vars] = scaler.transform(val_data_scaled[continuous_vars])  # 使用训练集的标准化参数
test_data_scaled[continuous_vars] = scaler.transform(test_data_scaled[continuous_vars])
# 保存标准化后的验证集和测试集
val_data_scaled.to_csv("outputs/data_processed/validation_scaled.csv")
test_data_scaled.to_csv("outputs/data_processed/test_scaled.csv")

#################### 基于训练集进行变量筛选（纳入到后续的Cox预测模型） ####################
# 1. 读取训练数据
train_data_notscaled = pd.read_csv("outputs/data_splits/train_raw.csv", index_col=0)  # 使用未标准化的训练集
# 2. 定义协变量
independent_vars = ['horTh', 'age', 'menostat', 'tsize', 'tgrade', 'pnodes', 'progrec', 'estrec']
## 3. 单因素Cox回归 
results_univariable = []
for var in independent_vars:
    print(f"###### {var} ######")
    df_temp = train_data_notscaled[[var, 'time', 'cens']].copy()
    cph = CoxPHFitter()
    cph.fit(df_temp, duration_col="time", event_col="cens")
    summary = cph.summary
    coef = summary.loc[var, "coef"]
    p = summary.loc[var, "p"]
    HR = summary.loc[var, "exp(coef)"]
    CI_lower = summary.loc[var, "exp(coef) lower 95%"]
    CI_upper = summary.loc[var, "exp(coef) upper 95%"]
    results_univariable.append({
        "Variable": var,
        "Coefficient": coef,
        "Hazard Ratio": HR,
        "95% CI Lower": CI_lower,
        "95% CI Upper": CI_upper,
        "P-value": p
    })
# 4. 转为DataFrame输出结果
results_univariable_df = pd.DataFrame(results_univariable)
print(results_univariable_df)
# 5. 提取 P < 0.1 的变量名
significant_vars_univ = results_univariable_df[results_univariable_df["P-value"] < 0.1]["Variable"].tolist()
print("单因素显著变量（P < 0.1）:", significant_vars_univ)
# 6. 多因素Cox回归 
df_multi = train_data_notscaled[significant_vars_univ + ['time', 'cens']].copy()
cph_multi = CoxPHFitter()
cph_multi.fit(df_multi, duration_col="time", event_col="cens")
# 7. 提取多因素分析结果
summary_multi = cph_multi.summary.reset_index()
summary_multi = summary_multi.rename(columns={
    "covariate": "Variable",
    "coef": "Coefficient",
    "exp(coef)": "Hazard Ratio",
    "exp(coef) lower 95%": "95% CI Lower",
    "exp(coef) upper 95%": "95% CI Upper",
    "p": "P-value"
})
summary_multi = summary_multi[["Variable", "Coefficient", "Hazard Ratio", "95% CI Lower", "95% CI Upper", "P-value"]]
print(summary_multi)
# 8. 保存显著变量（P<0.05）
significant_vars_multi = summary_multi[summary_multi["P-value"] < 0.05]["Variable"].tolist()
print("多因素显著变量（P < 0.05）:", significant_vars_multi)
# 9. 保存分析结果和多因素显著变量名
results_univariable_df.to_csv("outputs/tables/results_univariable_df.csv", index=False)  # 保存单因素Cox回归结果
summary_multi.to_csv("outputs/tables/results_mulvariable_df.csv", index=False)  # 保存多因素Cox回归结果
with open("outputs/significant_vars.pkl", "wb") as f:  # 保存多因素显著变量名，方便之后读取
    pickle.dump(significant_vars_multi, f)

###################################################################################################################
########### 基于训练集构建预测模型并基于验证集调优模型超参数（测试集数据仅用于最终独立测试） ############
###################################################################################################################
############################# 1. Cox模型 #############################
# 我们这里用多因素Cox回归分析显著的变量训练模型 #
# 1.1. 读取训练数据
train_data_notscaled = pd.read_csv("outputs/data_splits/train_raw.csv", index_col=0)  # 使用未标准化的训练集
# 1.2. 加载筛选后的多因素显著变量
with open("outputs/significant_vars.pkl", "rb") as f:
    significant_vars = pickle.load(f)
# 1.3. 构建训练数据
train_data_cox = train_data_notscaled[significant_vars + ['time', 'cens']].copy()
# 1.4. 拟合Cox模型
model_cox = CoxPHFitter()
model_cox.fit(train_data_cox, duration_col="time", event_col="cens")
# 1.5. 输出训练集的C-index
c_index_train_cox = model_cox.concordance_index_
print(f"训练集的C-index为：{c_index_train_cox:.3f}")
# 1.6. 计算验证集的C-index，与其他模型在相同验证集上比较
val_data_notscaled = pd.read_csv("outputs/data_splits/validation_raw.csv", index_col=0)  # 验证集数据
val_data_cox = val_data_notscaled[significant_vars + ['time', 'cens']].copy()
y_val_pred_cox = -model_cox.predict_partial_hazard(val_data_cox)  # lifelines的C-index要求较大得分对应较长生存时间
c_index_val_cox = concordance_index(val_data_cox['time'], y_val_pred_cox, val_data_cox['cens'])
print(f"验证集的C-index为：{c_index_val_cox:.3f}")
# 1.7. 保存模型
os.makedirs("outputs/models", exist_ok=True)  # 创建模型输出目录
with open("outputs/models/model_cox.pkl", "wb") as f:
    pickle.dump(model_cox, f)

####################### 2. 正则化Cox模型（使用所有变量） #######################
# 读取标准化后的训练数据和验证数据
train_data = pd.read_csv("outputs/data_processed/train_scaled.csv", index_col=0)
val_data_scaled = pd.read_csv("outputs/data_processed/validation_scaled.csv", index_col=0)
# 2.1. 构造生存对象 y 和协变量 X
X = train_data.drop(columns=["time", "cens"])
y = np.array(
    [(bool(c), t) for c, t in zip(train_data["cens"], train_data["time"])],
    dtype=[("cens", "bool"), ("time", "float")]
)
# 2.2. 使用已经划分好的训练集和验证集
X_train, y_train = X, y
X_val = val_data_scaled.drop(columns=["time", "cens"])
y_val = np.array(
    [(bool(c), t) for c, t in zip(val_data_scaled["cens"], val_data_scaled["time"])],
    dtype=[("cens", "bool"), ("time", "float")]
)
# 2.3. 训练默认参数正则化 Cox 模型（其实默认训练的是一组模型，验证的时候是默认用最小alpha的模型）
penaltycox_default = CoxnetSurvivalAnalysis()  # 默认是弹性网Cox  alphas: set automatically, l1_ratio: float = 0.5
penaltycox_default.fit(X_train, y_train)
# 查看模型默认参数
print("模型默认参数:", pd.DataFrame.from_dict(penaltycox_default.get_params(),orient='index'))
# 查看alpha最小时的模型的变量系数（该模型是用于验证集预测的模型）
coefs_minalpha_df = pd.DataFrame({
    'var_name' : X.columns,
    'var_coef' : penaltycox_default.coef_[:,-1],
    'alpha_' : penaltycox_default.alphas_[-1]
})
# 查看变量系数信息
print(coefs_minalpha_df)
# 2.4. 计算默认参数模型的验证集C-index
y_val_pred_penaltycox = penaltycox_default.predict(X_val)
cindex_penaltycox = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred_penaltycox)[0]
print(f"默认参数模型的验证集C-index: {cindex_penaltycox:.3f}")
# 进行超参数网格搜索调优（基于验证集 C-index 最高） #
# 2.5. 定义超参数搜索范围
param_grid = {
    'alphas': np.logspace(-4, 0, 20),  # 正则化强度
    'l1_ratios': [0.0001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # L1正则项比例，0.0001时接近Ridge-Cox，1时为Lasso-Cox
}
# 2.6. 使用网格搜索同时优化这些超参数
best_cindex_penaltycox = 0
penaltycox_model_best = None
best_params_penaltycox = {}
# 遍历所有超参数组合
for l1 in param_grid['l1_ratios']:
    for alpha in param_grid['alphas']:
        # 定义正则化Cox模型
        penaltycox_model = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=[alpha], fit_baseline_model=True)
        # 训练模型
        penaltycox_model.fit(X_train, y_train)
        # 计算验证集C-index
        y_val_pred = penaltycox_model.predict(X_val)
        cindex = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred)[0]
        # 记录最佳参数组合
        if cindex > best_cindex_penaltycox:
            best_cindex_penaltycox = cindex
            penaltycox_model_best = penaltycox_model
            best_params_penaltycox = {
                'l1_ratios': l1,
                'alphas': alpha
            }
# 2.7. 输出最优超参数和验证集预测效果
print("最佳 Penalty-Cox模型 的参数组合:", best_params_penaltycox)
print("默认参数 Penalty-Cox模型 的验证集 C-index:", cindex_penaltycox)
print("参数调优 Penalty-Cox模型 的验证集 C-index:", best_cindex_penaltycox)
print("调优 Penalty-Cox模型 的超参数:", pd.DataFrame.from_dict(penaltycox_model_best.get_params(), orient='index'))
# 2.8. 保存训练好的参数调优模型
with open("outputs/models/penaltycox_model.pkl", "wb") as f:
    pickle.dump(penaltycox_model_best, f)

############################# 3. 生存决策树模型 #############################
# 3.1. 使用默认参数训练生存决策树
survtree_default = SurvivalTree(random_state=123)
survtree_default.fit(X_train, y_train)
# 3.2. 验证集预测结果与 C-index
y_val_pred_survtree = survtree_default.predict(X_val)
cindex_survtree_default = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred_survtree)[0]
print(f"默认参数模型的验证集 C-index: {cindex_survtree_default:.3f}")
# 3.3. 定义超参数搜索范围
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, None], # 树的最大深度，None 表示不限深度
    "min_samples_split": [2, 5, 10, 15, 20], # 内部节点再划分所需最小样本数
    "min_samples_leaf": [1, 2, 4, 6, 8, 10] # 叶子节点最少样本数，避免过拟合
}
# 3.4. 网格搜索优化超参数
best_cindex_survtree = 0
survtree_model_best = None
best_params_survtree = {}
for depth in param_grid["max_depth"]:
    for min_split in param_grid["min_samples_split"]:
        for min_leaf in param_grid["min_samples_leaf"]:
            # 定义生存决策树模型
            survtree_model = SurvivalTree(
                max_depth=depth,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                random_state=123
            )
            # 训练模型
            survtree_model.fit(X_train, y_train)
            # 计算验证集C-index
            y_val_pred = survtree_model.predict(X_val)
            cindex = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred)[0]
            # 记录最佳参数组合
            if cindex > best_cindex_survtree:
                best_cindex_survtree = cindex
                survtree_model_best = survtree_model
                best_params_survtree = {
                    "max_depth": depth,
                    "min_samples_split": min_split,
                    "min_samples_leaf": min_leaf
                }
# 3.5. 输出调参结果
print("最佳生存决策树模型的参数组合:", best_params_survtree)
print("默认参数模型的验证集 C-index:", cindex_survtree_default)
print("调优后模型的验证集 C-index:", best_cindex_survtree)
print("调优生存决策树模型的超参数:", pd.DataFrame.from_dict(survtree_model_best.get_params(), orient='index'))
# 3.6. 保存模型
model_path = "outputs/models/survtree_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(survtree_model_best, f)
   
############################# 4. 随机生存森林模型 #############################
# 4.1. 使用默认参数训练随机生存森林
survrf_default = RandomSurvivalForest(random_state=123)
survrf_default.fit(X_train, y_train)
# 4.2. 验证集预测结果与 C-index
y_val_pred_survrf = survrf_default.predict(X_val)
cindex_survrf_default = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred_survrf)[0]
print(f"默认参数模型的验证集 C-index: {cindex_survrf_default:.3f}")
# 4.3. 定义超参数搜索范围
param_grid = {
    "n_estimators": [50, 100, 200, 300],               # 森林中树的数量
    "max_depth": [3, 5, 8, 10, None],                  # 每棵树的最大深度
    "min_samples_split": [2, 5, 10],                   # 内部节点再划分所需的最小样本数
    "min_samples_leaf": [1, 2, 4],                     # 叶子节点所需的最小样本数
    "max_features": ["sqrt", "log2", None]             # 每次分裂时考虑的特征数
}
# 4.4. 网格搜索优化超参数
best_cindex_survrf = 0
survrf_model_best = None
best_params_survrf = {}
for n_tree in param_grid["n_estimators"]:
    for depth in param_grid["max_depth"]:
        for min_split in param_grid["min_samples_split"]:
            for min_leaf in param_grid["min_samples_leaf"]:
                for max_feat in param_grid["max_features"]:
                    # 定义模型
                    survrf_model = RandomSurvivalForest(
                        n_estimators=n_tree,
                        max_depth=depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        max_features=max_feat,
                        random_state=123,
                        n_jobs=-1  # 并行计算加速训练
                    )
                    # 拟合模型
                    survrf_model.fit(X_train, y_train)
                    # 验证集C-index
                    y_val_pred = survrf_model.predict(X_val)
                    cindex = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred)[0]
                    # 更新最佳模型
                    if cindex > best_cindex_survrf:
                        best_cindex_survrf = cindex
                        survrf_model_best = survrf_model
                        best_params_survrf = {
                            "n_estimators": n_tree,
                            "max_depth": depth,
                            "min_samples_split": min_split,
                            "min_samples_leaf": min_leaf,
                            "max_features": max_feat
                        }
# 4.5. 输出调参结果
print("最佳随机生存森林模型的参数组合:", best_params_survrf)
print("默认参数模型的验证集 C-index:", cindex_survrf_default)
print("调优后模型的验证集 C-index:", best_cindex_survrf)
print("调优随机生存森林模型的超参数:", pd.DataFrame.from_dict(survrf_model_best.get_params(), orient='index'))
# 4.6. 保存模型
with open("outputs/models/survrf_model.pkl", "wb") as f:
    pickle.dump(survrf_model_best, f)

############################# 5. 生存梯度提升模型 #############################
# 5.1. 使用默认参数训练生存梯度提升模型
survgb_default = GradientBoostingSurvivalAnalysis(random_state=123)
survgb_default.fit(X_train, y_train)
# 5.2. 验证集预测结果与 C-index
y_val_pred_survgb = survgb_default.predict(X_val)
cindex_survgb_default = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred_survgb)[0]
print(f"默认参数模型的验证集 C-index: {cindex_survgb_default:.3f}")
# 5.3. 定义超参数搜索范围
param_grid = {
    "n_estimators": [50, 100, 200, 300],          # 弱分类器数量
    "learning_rate": [0.01, 0.05, 0.1, 0.2],      # 学习率（越小越稳定，越大越快）
    "max_depth": [2, 3, 4, 5, 6],                 # 每棵树的最大深度
    "min_samples_split": [2, 5, 10],              # 内部节点再划分所需的最小样本数
    "min_samples_leaf": [1, 2, 4]                 # 叶子节点最少样本数，避免过拟合
}
# 5.4. 网格搜索优化超参数
best_cindex_survgb = 0
survgb_model_best = None
best_params_survgb = {}
for n_tree in param_grid["n_estimators"]:
    for lr in param_grid["learning_rate"]:
        for depth in param_grid["max_depth"]:
            for min_split in param_grid["min_samples_split"]:
                for min_leaf in param_grid["min_samples_leaf"]:
                    # 定义模型
                    survgb_model = GradientBoostingSurvivalAnalysis(
                        n_estimators=n_tree,
                        learning_rate=lr,
                        max_depth=depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=123
                    )
                    # 拟合模型
                    survgb_model.fit(X_train, y_train)
                    # 验证集C-index
                    y_val_pred = survgb_model.predict(X_val)
                    cindex = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred)[0]
                    # 更新最佳模型
                    if cindex > best_cindex_survgb:
                        best_cindex_survgb = cindex
                        survgb_model_best = survgb_model
                        best_params_survgb = {
                            "n_estimators": n_tree,
                            "learning_rate": lr,
                            "max_depth": depth,
                            "min_samples_split": min_split,
                            "min_samples_leaf": min_leaf
                        }
# 5.5. 输出调参结果
print("最佳生存梯度提升模型的参数组合:", best_params_survgb)
print("默认参数模型的验证集 C-index:", cindex_survgb_default)
print("调优后模型的验证集 C-index:", best_cindex_survgb)
print("调优生存梯度提升模型的超参数:", pd.DataFrame.from_dict(survgb_model_best.get_params(), orient='index'))
# 5.6. 保存模型
with open("outputs/models/survgb_model.pkl", "wb") as f:
    pickle.dump(survgb_model_best, f)

############################# 6. 生存支持向量机模型 #############################
# 6.1. 使用默认参数训练生存SVM模型
survsvm_default = FastSurvivalSVM(random_state=123)
survsvm_default.fit(X_train, y_train)
# 6.2. 验证集预测结果与 C-index
y_val_pred_survsvm = survsvm_default.predict(X_val)
cindex_survsvm_default = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred_survsvm)[0]
print(f"默认参数模型的验证集 C-index: {cindex_survsvm_default:.3f}")
# 6.3. 定义超参数搜索范围
param_grid = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],     # 损失函数权重（越小正则化相对越强）
    "rank_ratio": [0.1, 0.2, 0.5, 0.8, 1.0]             # 排序与时间回归目标的权重比例，1.0为纯排序
}
# 6.4. 网格搜索优化超参数
best_cindex_survsvm = 0
survsvm_model_best = None
best_params_survsvm = {}

for a in param_grid["alpha"]:
    for rr in param_grid["rank_ratio"]:
            # 定义模型
            survsvm_model = FastSurvivalSVM(
                alpha=a,
                rank_ratio=rr,
                random_state=123
            )
            # 拟合模型
            survsvm_model.fit(X_train, y_train)
            # 验证集C-index（rank_ratio<1时预测的是生存时间，需转换为风险排序方向）
            y_val_pred = survsvm_model.predict(X_val)
            if rr < 1.0:
                y_val_pred = -y_val_pred
            cindex = concordance_index_censored(y_val["cens"], y_val["time"], y_val_pred)[0]
            # 更新最佳模型
            if cindex > best_cindex_survsvm:
                best_cindex_survsvm = cindex
                survsvm_model_best = survsvm_model
                best_params_survsvm = {
                    "alpha": a,
                    "rank_ratio": rr
                }
# 6.5. 输出调参结果
print("最佳生存支持向量机模型的参数组合:", best_params_survsvm)
print("默认参数模型的验证集 C-index:", cindex_survsvm_default)
print("调优后模型的验证集 C-index:", best_cindex_survsvm)
print("调优生存SVM模型的超参数:", pd.DataFrame.from_dict(survsvm_model_best.get_params(), orient='index'))
# 6.6. 保存模型
with open("outputs/models/survsvm_model.pkl", "wb") as f:
    pickle.dump(survsvm_model_best, f)

################################################################################################
################################## 验证数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
# 构建训练集和验证集（对于Cox模型）
with open('outputs/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)
significant_vars_multi
train_data_notscaled = pd.read_csv("outputs/data_splits/train_raw.csv", index_col=0)
val_data_notscaled = pd.read_csv("outputs/data_splits/validation_raw.csv", index_col=0)
train_data_cox = train_data_notscaled[significant_vars_multi + ['time', 'cens']].copy()
val_data_cox = val_data_notscaled[significant_vars_multi + ['time', 'cens']].copy()
# 构造训练集和验证集（对于机器学习生存模型）
train_data_scaled = pd.read_csv("outputs/data_processed/train_scaled.csv", index_col=0)
val_data_scaled = pd.read_csv("outputs/data_processed/validation_scaled.csv", index_col=0)
X_train = train_data_scaled.drop(columns=["time", "cens"])
X_val = val_data_scaled.drop(columns=["time", "cens"])
y_train = np.array(
    [(bool(c), t) for c, t in zip(train_data_scaled["cens"], train_data_scaled["time"])],
    dtype=[("cens", "bool"), ("time", "float")])
y_val = np.array(
    [(bool(c), t) for c, t in zip(val_data_scaled["cens"], val_data_scaled["time"])],
    dtype=[("cens", "bool"), ("time", "float")])
# 使用训练集估计删失分布，后续动态AUC、DCA和ROC均不使用验证集拟合删失权重
from sksurv.nonparametric import CensoringDistributionEstimator
censor_model = CensoringDistributionEstimator().fit(y_train)
# 固定时间点预测主要展示3年：验证集3年风险集72人、5年仅24人；5年可作为探索性评价
validation_time_point = 365 * 3
os.makedirs("outputs/figures/evaluation", exist_ok=True)

###################### 加载训练好的模型 ##########################
# Cox比例风险模型
with open("outputs/models/model_cox.pkl", "rb") as f:
    model_cox = pickle.load(f)
# 正则化Cox模型（Coxnet）
with open("outputs/models/penaltycox_model.pkl", "rb") as f:
    penaltycox_model = pickle.load(f)
# 生存决策树模型
with open("outputs/models/survtree_model.pkl", "rb") as f:
    survtree_model = pickle.load(f)
# 随机生存森林模型
with open("outputs/models/survrf_model.pkl", "rb") as f:
    survrf_model = pickle.load(f)
# 生存梯度提升模型（GBM）
with open("outputs/models/survgb_model.pkl", "rb") as f:
    survgb_model = pickle.load(f)
# 生存支持向量机模型（SVM）
with open("outputs/models/survsvm_model.pkl", "rb") as f:
    survsvm_model = pickle.load(f)
## run end ##

############################## 1. C-index 及其95%置信区间 ##################################
# 获取模型风险得分（统一为得分越大、事件风险越高）
def get_risk_score(model, X, model_type="Cox"):
    if model_type == "Cox":
        pred_score = model.predict_partial_hazard(X)
    elif model_type in ("Machine Survival", "ML Survival", "SVM"):
        pred_score = model.predict(X)
        if isinstance(model, FastSurvivalSVM) and model.rank_ratio < 1.0:
            pred_score = -pred_score  # rank_ratio<1的SVM输出生存时间，方向需取反
    else:
        raise ValueError("model_type must be 'Cox' or 'Machine Survival'")
    return np.asarray(pred_score, dtype=float).reshape(-1)

# 定义C-index及其Bootstrapping置信区间的计算函数
def bootstrap_cindex_survmodel(model, X, y, model_type="Cox", n_bootstrap=500, random_state=123):
    """
    计算生存模型的C-index及其95%置信区间（Bootstrapping法）
    参数说明：
    - model: 已训练好的模型（Cox模型或机器学习生存模型）
    - X: 特征数据（DataFrame或ndarray）
    - y: 结构化生存对象（sksurv）或包含'time'和'cens'的DataFrame
    - model_type: "Cox" 或 "Machine Survival"
    - n_bootstrap: Bootstrap迭代次数
    - random_state: 随机种子
    返回值：
    - cindex: C-index的点估计
    - cindex_ci_lower: 2.5%分位数
    - cindex_ci_upper: 97.5%分位数
    """
    # 统一在验证集计算各模型的C-index
    pred_score = get_risk_score(model, X, model_type=model_type)
    if model_type == "Cox":
        event, time = X["cens"].to_numpy(dtype=bool), X["time"].to_numpy(dtype=float)
    else:
        event, time = y["cens"].astype(bool), y["time"].astype(float)
    cindex = concordance_index_censored(event, time, pred_score)[0]
    # 患者级Bootstrap：重抽样验证集患者及其固定预测，不重新训练模型
    rng = np.random.default_rng(random_state)
    cindexes_list = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(pred_score), size=len(pred_score), replace=True)
        try:
            cindex_boot = concordance_index_censored(event[indices], time[indices], pred_score[indices])[0]
            cindexes_list.append(cindex_boot)
        except ValueError:  # 极少数重抽样可能不存在可比较的事件对
            continue
    # 计算Bootstrap百分位数95%置信区间（验证集区间仅作描述性参考）
    cindex_ci_lower, cindex_ci_upper = np.percentile(cindexes_list, [2.5, 97.5])
    return cindex, cindex_ci_lower, cindex_ci_upper

# 1.1. Cox比例风险模型（使用与其他模型相同的验证集）
cindex_cox_valid, cindex_ci_lower_cox_valid, cindex_ci_upper_cox_valid = \
    bootstrap_cindex_survmodel(model=model_cox, X=val_data_cox, y=None, model_type="Cox")
print(f"Cox模型验证集 C-index: {cindex_cox_valid:.4f} (95% CI: [{cindex_ci_lower_cox_valid:.4f}, {cindex_ci_upper_cox_valid:.4f}])")
# 1.2. 正则化Cox模型（Coxnet）
cindex_penaltycox_valid, ci_lower_penaltycox_valid, ci_upper_penaltycox_valid = \
    bootstrap_cindex_survmodel(model=penaltycox_model, X=X_val, y=y_val, model_type="Machine Survival")
print(f"正则化Cox模型验证集 C-index: {cindex_penaltycox_valid:.4f} (95% CI: [{ci_lower_penaltycox_valid:.4f}, {ci_upper_penaltycox_valid:.4f}])")
# 1.3. 生存决策树模型
cindex_survtree_valid, ci_lower_survtree_valid, ci_upper_survtree_valid = \
    bootstrap_cindex_survmodel(model=survtree_model, X=X_val, y=y_val, model_type="Machine Survival")
print(f"生存决策树模型验证集 C-index: {cindex_survtree_valid:.4f} (95% CI: [{ci_lower_survtree_valid:.4f}, {ci_upper_survtree_valid:.4f}])")
# 1.4. 随机生存森林模型
cindex_survrf_valid, ci_lower_survrf_valid, ci_upper_survrf_valid = \
    bootstrap_cindex_survmodel(model=survrf_model, X=X_val, y=y_val, model_type="Machine Survival")
print(f"随机生存森林模型验证集 C-index: {cindex_survrf_valid:.4f} (95% CI: [{ci_lower_survrf_valid:.4f}, {ci_upper_survrf_valid:.4f}])")
# 1.5. 生存梯度提升模型
cindex_survgb_valid, ci_lower_survgb_valid, ci_upper_survgb_valid = \
    bootstrap_cindex_survmodel(model=survgb_model, X=X_val, y=y_val, model_type="Machine Survival")
print(f"生存梯度提升模型验证集 C-index: {cindex_survgb_valid:.4f} (95% CI: [{ci_lower_survgb_valid:.4f}, {ci_upper_survgb_valid:.4f}])")
# 1.6. 生存支持向量机模型
cindex_survsvm_valid, ci_lower_survsvm_valid, ci_upper_survsvm_valid = \
    bootstrap_cindex_survmodel(model=survsvm_model, X=X_val, y=y_val, model_type="Machine Survival")
print(f"生存支持向量机模型验证集 C-index: {cindex_survsvm_valid:.4f} (95% CI: [{ci_lower_survsvm_valid:.4f}, {ci_upper_survsvm_valid:.4f}])")
# 保存这些模型的验证集C-index结果 #
df_cindex_valid = pd.DataFrame({
    "模型名称": ["Cox模型", "正则化Cox模型", "生存决策树模型", "随机生存森林模型", "生存梯度提升模型", "生存支持向量机模型"],
    "C-index": [cindex_cox_valid, cindex_penaltycox_valid, cindex_survtree_valid, cindex_survrf_valid, cindex_survgb_valid, cindex_survsvm_valid],
    "95%CI下限": [cindex_ci_lower_cox_valid, ci_lower_penaltycox_valid, ci_lower_survtree_valid, ci_lower_survrf_valid, ci_lower_survgb_valid, ci_lower_survsvm_valid],
    "95%CI上限": [cindex_ci_upper_cox_valid, ci_upper_penaltycox_valid, ci_upper_survtree_valid, ci_upper_survrf_valid, ci_upper_survgb_valid, ci_upper_survsvm_valid]
})
df_cindex_valid.to_csv("outputs/tables/model_cindex_validation.csv", index=False, encoding="utf-8-sig")

###################### 2. 时间依赖性AUC（Time-dependent AUC）及绘制其折线图 ##########################
# 用直方图查看生存时间大致分布，帮助确定要计算AUC的时间点
plt.figure(figsize=(8, 5))
plt.hist(train_data_cox['time'], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Time (days)")
plt.ylabel("Frequency")
plt.title("Histogram of Observed Follow-up Time")
plt.tight_layout()
plt.savefig("outputs/figures/evaluation/observed_time_histogram.png", dpi=300)
plt.show()
# 绘制时间依赖性AUC折线图的函数
def plot_time_dependent_auc(model, X, y_struct, eval_times, label="Model", model_type="Cox"):
    """
    绘制时间依赖性AUC曲线

    参数说明：
    - model: 拟合好的模型
    - X: 用于预测的特征数据（DataFrame或ndarray）
    - y_struct: 验证集/测试集结构化生存对象（dtype=[('cens', bool), ('time', float)]）
    - eval_times: 时间点数组（单位与time一致，如天）
    - label: 图中显示的模型名称
    - model_type: "Cox" 或 "Machine Survival"
    """
    # 获取预测得分（越大表示事件风险越高）
    pred_score = get_risk_score(model, X, model_type=model_type)
    # 训练集估计删失分布，验证集评价；不再将y_struct同时用作训练集和评价集
    auc_vals, mean_auc = cumulative_dynamic_auc(y_train, y_struct, pred_score, times=eval_times)
    print(f"{label} 平均时间依赖AUC: {mean_auc:.4f}")
    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(eval_times / (365 / 12), auc_vals, marker='o', linewidth=1, label=f"{label}")
    plt.xlabel("Time (months)")
    plt.ylabel("Time-dependent AUC")
    plt.title("Time-dependent AUC Curve")
    plt.ylim(0., 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/auc_{label.lower().replace(' ', '_')}_validation.png", dpi=300)
    plt.show()
    return(mean_auc, auc_vals)

## 2.1. Cox比例风险模型 ##
# 验证集的结构化结局必须是cens=1表示事件，不能反转编码
# 原代码展示6年AUC时末期风险集很少；此处保留6个月至5年的评价时点
eval_times = np.array([180, 270, 365, 365*1.5, 365*2, 365*2.5, 365*3, 365*3.5,
                       365*4, 365*4.5, 365*5])
mean_auc_cox_valid, auc_valus_cox_valid = plot_time_dependent_auc(model=model_cox, X=val_data_cox,
    y_struct=y_val, eval_times=eval_times, label="Cox Model", model_type="Cox")
mean_auc_cox_valid, auc_valus_cox_valid
# 2.2. 正则化Cox模型
mean_auc_penaltycox_valid, auc_valus_penaltycox_valid = plot_time_dependent_auc(model=penaltycox_model, X=X_val,
    y_struct=y_val, eval_times=eval_times, label="Regularized Cox Model", model_type="Machine Survival")
mean_auc_penaltycox_valid, auc_valus_penaltycox_valid
# 2.3. 生存决策树模型
mean_auc_survtree_valid, auc_valus_survtree_valid = plot_time_dependent_auc(model=survtree_model, X=X_val,
    y_struct=y_val, eval_times=eval_times, label="Survival Tree", model_type="Machine Survival")
mean_auc_survtree_valid, auc_valus_survtree_valid
# 2.4. 随机生存森林模型
mean_auc_survrf_valid, auc_valus_survrf_valid = plot_time_dependent_auc(model=survrf_model, X=X_val,
    y_struct=y_val, eval_times=eval_times, label="Random Survival Forest", model_type="Machine Survival")
mean_auc_survrf_valid, auc_valus_survrf_valid
# 2.5. 生存梯度提升模型
mean_auc_survgb_valid, auc_valus_survgb_valid = plot_time_dependent_auc(model=survgb_model, X=X_val,
    y_struct=y_val, eval_times=eval_times, label="Gradient Boosting Survival", model_type="Machine Survival")
mean_auc_survgb_valid, auc_valus_survgb_valid
# 2.6. 生存支持向量机模型
mean_auc_survsvm_valid, auc_valus_survsvm_valid = plot_time_dependent_auc(model=survsvm_model, X=X_val,
    y_struct=y_val, eval_times=eval_times, label="Survival SVM", model_type="Machine Survival")
mean_auc_survsvm_valid, auc_valus_survsvm_valid
# 多模型验证集的时间依赖AUC对比折线图 #
plt.figure(figsize=(10, 6))
plt.plot(eval_times / (365 / 12), auc_valus_cox_valid, marker='o', label="Cox Model")
plt.plot(eval_times / (365 / 12), auc_valus_penaltycox_valid, marker='o', label="Regularized Cox Model")
plt.plot(eval_times / (365 / 12), auc_valus_survtree_valid, marker='o', label="Survival Tree")
plt.plot(eval_times / (365 / 12), auc_valus_survrf_valid, marker='o', label="Random Survival Forest")
plt.plot(eval_times / (365 / 12), auc_valus_survgb_valid, marker='o', label="Gradient Boosting Survival")
plt.plot(eval_times / (365 / 12), auc_valus_survsvm_valid, marker='o', label="Survival SVM")
plt.xlabel("Time (months)")
plt.ylabel("Time-dependent AUC")
plt.title("Time-dependent AUC Comparison (Validation Set)")
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/evaluation/AUC_curves_allmodel_validation.png", dpi=300)
plt.show()
# 时间依赖AUC估计值整理保存为CSV表格（验证集）#
df_auc_valid = pd.DataFrame({
    "评估时间（月）": np.insert(np.round(eval_times / (365 / 12), 2).astype(str), 0, "平均AUC"),
    "Cox模型": np.insert(auc_valus_cox_valid, 0, mean_auc_cox_valid),
    "正则化Cox模型": np.insert(auc_valus_penaltycox_valid, 0, mean_auc_penaltycox_valid),
    "生存决策树模型": np.insert(auc_valus_survtree_valid, 0, mean_auc_survtree_valid),
    "随机生存森林模型": np.insert(auc_valus_survrf_valid, 0, mean_auc_survrf_valid),
    "生存梯度提升模型": np.insert(auc_valus_survgb_valid, 0, mean_auc_survgb_valid),
    "生存支持向量机模型": np.insert(auc_valus_survsvm_valid, 0, mean_auc_survsvm_valid)
})
df_auc_valid.to_csv("outputs/tables/model_time_auc_validation.csv", index=False, encoding="utf-8-sig")

#################################### 3. 绘制某个指定时间点的校准曲线 ####################################
# 校准曲线比较模型预测的生存概率与各风险组KM估计的实际生存概率
def plot_calibration_points(pred_surv_probs, time, event, time_point, n_groups, label):
    df_calib = pd.DataFrame({"pred_surv": pred_surv_probs, "time": time, "event": event})
    df_calib["group"] = pd.qcut(df_calib["pred_surv"], n_groups, labels=False, duplicates='drop')
    observed_surv, predicted_surv = [], []
    kmf = KaplanMeierFitter()
    for i in range(df_calib["group"].nunique()):
        group_data = df_calib[df_calib["group"] == i]
        kmf.fit(durations=group_data["time"], event_observed=group_data["event"])
        # 组内随访不支持该时间点时不外推KM曲线
        if group_data["time"].max() < time_point and kmf.survival_function_.iloc[-1, 0] > 0:
            print(f"{label}: 分组{i}在目标时间之前结束观察，跳过该组校准点")
            continue
        observed_surv.append(float(kmf.predict(time_point)))
        predicted_surv.append(group_data["pred_surv"].mean())
    if len(observed_surv) < 2:
        print(f"{label}: 有效校准分组不足，跳过校准曲线")
        return
    plt.figure(figsize=(8, 6))
    plt.plot(predicted_surv, observed_surv, marker='o', linestyle='-', label=f"{label} Calibration")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    plt.xlabel("Predicted Survival Probability")
    plt.ylabel("Observed Survival Probability (KM)")
    plt.title(f"Calibration Curve at {time_point / 365:g} Years")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/calibration_{label.lower().replace(' ', '_')}_validation.png", dpi=300)
    plt.show()

# 绘制校准曲线的函数（适用于Cox模型）
def plot_calibration_curve_cox(model, data, time_col="time", event_col="cens",
                               label="Cox Model", time_point=365*3, n_groups=3):
    # 在目标时间点直接计算生存概率，而不是选择最接近的模型时间点
    surv_func_df = model.predict_survival_function(data.drop(columns=[time_col, event_col]), times=[time_point])
    pred_surv_probs = surv_func_df.iloc[0, :].to_numpy()
    plot_calibration_points(pred_surv_probs, data[time_col].to_numpy(), data[event_col].to_numpy(),
                            time_point, n_groups, label)

# 绘制校准曲线的函数（适用于机器学习模型）#
def plot_calibration_curve_ml(model, X, y, time_point=365*3, n_groups=3, label="ML Survival Model"):
    if not hasattr(model, "predict_survival_function"):
        print(f"模型 {label} 不支持 survival function 输出，跳过校准曲线绘制。") # 自动跳过不支持校准的模型
        return
    surv_funcs = model.predict_survival_function(X)
    pred_surv_probs = np.array([fn(time_point) for fn in surv_funcs])
    plot_calibration_points(pred_surv_probs, y["time"], y["cens"], time_point, n_groups, label)

# 3.1. Cox模型
plot_calibration_curve_cox(model=model_cox, data=val_data_cox,
                           n_groups=3, label="Cox Model", time_point=validation_time_point)
# 3.2. 正则化Cox模型
plot_calibration_curve_ml(model=penaltycox_model, X=X_val, y=y_val,
                          n_groups=3, label="Regularized Cox Model", time_point=validation_time_point)
# 出现预测生存概率集中的现象，可能与模型输出分布、样本量和删失较多有关
# 3.3. 生存决策树模型
plot_calibration_curve_ml(model=survtree_model, X=X_val, y=y_val,
                          n_groups=3, label="Survival Tree Model", time_point=validation_time_point)
# 3.4. 随机生存森林模型
plot_calibration_curve_ml(model=survrf_model, X=X_val, y=y_val,
                          n_groups=3, label="Random Survival Forest", time_point=validation_time_point)
# 3.5. 生存梯度提升模型
plot_calibration_curve_ml(model=survgb_model, X=X_val, y=y_val,
                          n_groups=3, label="Gradient Boosting Survival", time_point=validation_time_point)
# 3.6. 生存支持向量机模型
plot_calibration_curve_ml(model=survsvm_model, X=X_val, y=y_val,
                          n_groups=3, label="Survival SVM", time_point=validation_time_point)
    # FastSurvivalSVM 是基于 ranking-based 支持向量方法，它优化的是“排序”，不是生存函数、也不是生存概率，最终输出的是一个 risk score（风险得分），只能用于排序、计算 C-index。
    # scikit-survival 中的 SVM 类模型目前都不支持输出生存函数或生存概率 S(t)。它们全部是为了 风险排序或边界优化 设计的，而不是拟合 S(t)。

###################### 4. 绘制决策分析曲线 ##########################
# 定义计算不同阈值下净收益的函数（IPCW处理右删失）#
def calculate_net_benefit_survival(time, event, pred_surv_probs,
                                   time_point=365*3,
                                   thresholds=np.linspace(0.01, 0.99, 100)):
    pred_risk_probs = 1 - np.asarray(pred_surv_probs)
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    n = len(time)
    # 在训练集估计的删失分布上获取IPCW权重
    y_eval = np.array(list(zip(event, time)), dtype=[("cens", bool), ("time", float)])
    event_weights = censor_model.predict_ipcw(y_eval)
    censor_surv = censor_model.predict_proba(np.array([time_point]))[0]
    if censor_surv <= 0:
        raise ValueError("目标时间点的删失生存概率为0，无法计算IPCW DCA")
    # 在目标时间点之前已发生事件=病例；观察超过目标时间点=对照
    # 提前右删失者此时结局未知，不直接当作阴性
    case_weight = ((event == 1) & (time <= time_point)) * event_weights
    control_weight = (time > time_point).astype(float) / censor_surv
    net_benefit_model, net_benefit_all, net_benefit_none = [], [], [0] * len(thresholds)
    tp_all, fp_all = case_weight.sum() / n, control_weight.sum() / n
    for thresh in thresholds:
        y_pred = pred_risk_probs > thresh
        tp = np.sum(y_pred * case_weight) / n
        fp = np.sum(y_pred * control_weight) / n
        net_benefit_model.append(tp - fp * (thresh / (1 - thresh)))
        net_benefit_all.append(tp_all - fp_all * (thresh / (1 - thresh)))
    return net_benefit_model, net_benefit_all, net_benefit_none

# 定义DCA绘图函数 #
def plot_dca_curve(thresholds, net_benefit_model, net_benefit_all, net_benefit_none, label="Model"):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model, label=label, color="blue", linewidth=2)
    plt.plot(thresholds, net_benefit_all, label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefit_none, label="Treat None", color="green", linestyle="--", linewidth=2)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.ylim(-0.15, np.nanmax(np.array(net_benefit_model)) + 0.1)
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/dca_{label.lower().replace(' ', '_')}_validation.png", dpi=300)
    plt.show()

# 定义净收益计算和DCA绘图函数 #
def plot_dca_survival(model, X, time, event, time_point=365*3,
                      label="Model", model_type="Cox", thresholds=np.linspace(0.01, 0.99, 100)):
    """
    通用DCA绘图函数，适用于 lifelines Cox模型和 scikit-survival 生存机器学习模型
    参数：
    - model: 拟合好的模型
    - X: 特征数据（DataFrame或ndarray）
    - time: 生存时间（array-like）
    - event: 事件状态（1=事件，0=删失）
    - time_point: 指定预测的时间点（如365*3表示3年）
    - label: 图例标签
    - model_type: "Cox" 或 "ML Survival"
    - thresholds: 阈值列表
    """
    # 获取预测的生存概率
    if model_type == "Cox":
        surv_funcs = model.predict_survival_function(X.drop(columns=["time", "cens"]), times=[time_point])
        pred_surv_probs = surv_funcs.iloc[0, :].to_numpy()
    elif model_type == "ML Survival":
        surv_funcs = model.predict_survival_function(X)
        pred_surv_probs = np.array([fn(time_point) for fn in surv_funcs])
    else:
        raise ValueError("model_type must be either 'Cox' or 'ML Survival'")
    # 计算净收益
    net_benefit_model, net_benefit_all, net_benefit_none = calculate_net_benefit_survival(
        time=time, event=event, pred_surv_probs=pred_surv_probs, time_point=time_point, thresholds=thresholds
    )
    # 绘制DCA曲线
    plot_dca_curve(thresholds, net_benefit_model, net_benefit_all, net_benefit_none, label=label)

# 4.1. Cox 模型
plot_dca_survival(model=model_cox, X=val_data_cox, time=val_data_cox["time"].values,
                  event=val_data_cox["cens"].values, time_point=validation_time_point,
                  label="Cox Model", model_type="Cox")
# 4.2. 正则化 Cox 模型
plot_dca_survival(model=penaltycox_model, X=X_val, time=y_val["time"],
                  event=y_val["cens"].astype(int), time_point=validation_time_point,
                  label="Regularized Cox", model_type="ML Survival")
# 4.3. 生存决策树模型
plot_dca_survival(model=survtree_model, X=X_val, time=y_val["time"],
                  event=y_val["cens"].astype(int), time_point=validation_time_point,
                  label="Survival Tree", model_type="ML Survival")
# 4.4. 随机生存森林模型
plot_dca_survival(model=survrf_model, X=X_val, time=y_val["time"],
                  event=y_val["cens"].astype(int), time_point=validation_time_point,
                  label="Random Survival Forest", model_type="ML Survival")
# 4.5. 生存梯度提升模型
plot_dca_survival(model=survgb_model, X=X_val, time=y_val["time"],
                  event=y_val["cens"].astype(int), time_point=validation_time_point,
                  label="Gradient Boosting Survival", model_type="ML Survival")
## 生存支持向量机模型不支持生存率的计算，所以同样无法绘制DCA

# 各模型3年DCA曲线对比汇总图（不含SVM）#
time_point = validation_time_point
thresholds = np.linspace(0.01, 0.99, 100)
# 计算各模型3年预测生存概率
surv_cox = model_cox.predict_survival_function(val_data_cox.drop(columns=["time", "cens"]), times=[time_point])
p_cox = surv_cox.iloc[0, :].to_numpy()
p_penaltycox = np.array([fn(time_point) for fn in penaltycox_model.predict_survival_function(X_val)])
p_tree = np.array([fn(time_point) for fn in survtree_model.predict_survival_function(X_val)])
p_rsf = np.array([fn(time_point) for fn in survrf_model.predict_survival_function(X_val)])
p_gbs = np.array([fn(time_point) for fn in survgb_model.predict_survival_function(X_val)])
# 计算各模型净收益
nb_cox, nb_all, nb_none = calculate_net_benefit_survival(y_val["time"], y_val["cens"], p_cox, time_point, thresholds)
nb_penaltycox, _, _ = calculate_net_benefit_survival(y_val["time"], y_val["cens"], p_penaltycox, time_point, thresholds)
nb_tree, _, _ = calculate_net_benefit_survival(y_val["time"], y_val["cens"], p_tree, time_point, thresholds)
nb_rsf, _, _ = calculate_net_benefit_survival(y_val["time"], y_val["cens"], p_rsf, time_point, thresholds)
nb_gbs, _, _ = calculate_net_benefit_survival(y_val["time"], y_val["cens"], p_gbs, time_point, thresholds)
# 绘制决策分析曲线汇总图
plt.figure(figsize=(10, 6))
plt.plot(thresholds, nb_cox, label="Cox Model", linewidth=2)
plt.plot(thresholds, nb_penaltycox, label="Regularized Cox", linewidth=2)
plt.plot(thresholds, nb_tree, label="Survival Tree", linewidth=2)
plt.plot(thresholds, nb_rsf, label="Random Survival Forest", linewidth=2)
plt.plot(thresholds, nb_gbs, label="Gradient Boosting Survival", linewidth=2)
plt.plot(thresholds, nb_all, label="Treat All", color="red", linestyle="--")
plt.plot(thresholds, nb_none, label="Treat None", color="green", linestyle="--")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (3-Year, Validation Set)")
plt.ylim(-0.15, np.nanmax([nb_cox, nb_penaltycox, nb_tree, nb_rsf, nb_gbs]) + 0.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/figures/evaluation/DCA_curves_allmodel_validation.png", dpi=300)
plt.show()
# 各模型3年DCA净收益汇总（含最大值及其对应阈值、平均值）#
# 注意：最大值与平均值仅是对当前阈值范围的教学描述，不用于推荐临床决策阈值
def summarize_dca(nb_list, thresholds):
    nb_array = np.array(nb_list)
    max_nb = np.nanmax(nb_array)
    max_thresh = thresholds[np.nanargmax(nb_array)]
    mean_nb = np.nanmean(nb_array)
    return max_nb, max_thresh, mean_nb
summary = []
summary.append(summarize_dca(nb_all, thresholds))
summary.append(summarize_dca(nb_cox, thresholds))
summary.append(summarize_dca(nb_penaltycox, thresholds))
summary.append(summarize_dca(nb_tree, thresholds))
summary.append(summarize_dca(nb_rsf, thresholds))
summary.append(summarize_dca(nb_gbs, thresholds))
df_dca_full = pd.DataFrame(summary, columns=["最大净收益", "对应阈值", "平均净收益"])
df_dca_full.insert(0, "模型名称", ["治疗所有人", "Cox模型", "正则化Cox模型", "生存决策树模型", "随机生存森林模型", "生存梯度提升模型"])
df_dca_full.to_csv("outputs/tables/model_dca_3year_validation.csv", index=False, encoding="utf-8-sig")

###################### 5. 基于模型预测值分为低、中、高风险组，绘制KM生存曲线 ##########################
def plot_km_by_model_prediction(model, pred_data, time, event,
                                label="Model", model_type="Cox",
                                time_point=365*3, n_groups=3, reference_data=None):
    """
    根据模型预测值分组，绘制Kaplan-Meier生存曲线，并添加整体Log-rank检验P值

    参数说明：
    - model: 拟合好的生存模型
    - pred_data: 验证集/测试集的模型输入数据
    - time, event: 生存时间与结局状态（array-like）
    - label: 图标题和图例标签
    - model_type: "Cox"、"ML Survival" 或 "SVM"
    - time_point: 预测的目标时间点（仅用于生存概率计算）
    - n_groups: 风险分组数，默认3组（Low, Medium, High）
    - reference_data: 用于计算风险分组界值的训练集
    注意：
    - pred_score越大表示风险越高（score越高 → 生存概率越低）
    """
    if reference_data is None:
        reference_data = train_data_cox if model_type == "Cox" else X_train  # 后续测试集也沿用训练集界值
    # 1. 获取预测得分（越大风险越高），并用训练集确定分组阈值
    if model_type == "Cox":
        pred_score = get_risk_score(model, pred_data, model_type="Cox")
        train_score = get_risk_score(model, reference_data, model_type="Cox")
    elif model_type == "ML Survival":
        surv_funcs = model.predict_survival_function(pred_data)
        train_funcs = model.predict_survival_function(reference_data)
        pred_score = 1 - np.array([fn(time_point) for fn in surv_funcs])
        train_score = 1 - np.array([fn(time_point) for fn in train_funcs])
    elif model_type == "SVM":
        pred_score = get_risk_score(model, pred_data, model_type="SVM")
        train_score = get_risk_score(model, reference_data, model_type="SVM")
    else:
        raise ValueError("model_type must be 'Cox', 'ML Survival' or 'SVM'")
    # 2. 构建DataFrame并分组（训练集的三分位阈值应用于验证集）
    cuts = np.unique(np.quantile(train_score, np.arange(1, n_groups) / n_groups))
    group_labels = ["Low", "Medium", "High"] if len(cuts) == 2 else [f"Group {i+1}" for i in range(len(cuts)+1)]
    df_km = pd.DataFrame({"time": np.asarray(time), "event": np.asarray(event), "pred_score": pred_score})
    df_km["risk_group"] = pd.cut(df_km["pred_score"], bins=[-np.inf, *cuts, np.inf], labels=group_labels)
    # 3. 绘制KM曲线
    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    present_groups = []
    for group in group_labels:
        group_data = df_km[df_km["risk_group"] == group]
        if len(group_data) == 0:
            continue
        present_groups.append(group)
        kmf.fit(durations=group_data["time"], event_observed=group_data["event"], label=group)
        kmf.plot_survival_function(ci_show=False, linewidth=2)
    plt.title(f"Kaplan-Meier Survival Curve ({label})")
    plt.xlabel("Time (days)")
    plt.ylim(0, 1.02)
    plt.ylabel("Event-free Survival Probability")
    plt.legend(title="Risk Group")
    plt.grid(True)
    # 4. 整体Log-rank检验（多组比较，仅用于风险分层描述）
    if len(present_groups) > 1:
        results = multivariate_logrank_test(df_km["time"], df_km["risk_group"], event_observed=df_km["event"])
        p_value = results.p_value
        p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
        plt.text(0.05, 0.1, f"Log-rank P {p_str}", transform=plt.gca().transAxes,
                 horizontalalignment='left', fontsize=13,
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/KM_{label.lower().replace(' ', '_')}_validation.png", dpi=300)
    plt.show()

# 5.1. Cox模型
plot_km_by_model_prediction(model=model_cox, pred_data=val_data_cox,
                            time=val_data_cox["time"].values, event=val_data_cox["cens"].values,
                            label="Cox Model", model_type="Cox", reference_data=train_data_cox)
# 5.2. 正则化Cox模型
plot_km_by_model_prediction(model=penaltycox_model, pred_data=X_val,
                            time=y_val["time"], event=y_val["cens"].astype(int),
                            label="Regularized Cox Model", model_type="ML Survival",
                            time_point=validation_time_point, reference_data=X_train)
# 5.3. 生存决策树模型
plot_km_by_model_prediction(model=survtree_model, pred_data=X_val,
                            time=y_val["time"], event=y_val["cens"].astype(int),
                            label="Survival Tree", model_type="ML Survival",
                            time_point=validation_time_point, reference_data=X_train)
# 5.4. 随机生存森林模型
plot_km_by_model_prediction(model=survrf_model, pred_data=X_val,
                            time=y_val["time"], event=y_val["cens"].astype(int),
                            label="Random Survival Forest", model_type="ML Survival",
                            time_point=validation_time_point, reference_data=X_train)
# 5.5. 生存梯度提升模型
plot_km_by_model_prediction(model=survgb_model, pred_data=X_val,
                            time=y_val["time"], event=y_val["cens"].astype(int),
                            label="Gradient Boosting Survival", model_type="ML Survival",
                            time_point=validation_time_point, reference_data=X_train)
# 5.6. 生存支持向量机模型（直接用风险得分分组，不需要预测生存概率）
plot_km_by_model_prediction(model=survsvm_model, pred_data=X_val,
                            time=y_val["time"], event=y_val["cens"].astype(int),
                            label="Survival SVM", model_type="SVM", reference_data=X_train)

###################### 6. 绘制指定时间点的ROC曲线（IPCW校正右删失） ##########################
# 原始版本将生存问题简化为二分类并丢弃提前删失样本，可能产生选择偏倚；此处采用加权动态ROC
# 绘制指定时间点模型的ROC曲线的函数 #
def plot_survival_roc_by_model(model, pred_data, time, cens,
                               eval_time_point=365*3, label_time=None,
                               model_type="Cox"):
    """
    绘制指定时间点的生存ROC曲线（累计病例/动态对照，使用IPCW处理右删失）
    - pred_data: 用于预测的特征数据
    - time: 生存时间
    - cens: 事件状态（1=事件发生，0=右删失）
    - eval_time_point: 指定评估时间点
    - model_type: "Cox" 或 "Machine Survival"
    """
    # 1. 获取模型风险得分（较大表示更高事件风险）
    risk_score = get_risk_score(model, pred_data, model_type=model_type)
    time, event = np.asarray(time, dtype=float), np.asarray(cens, dtype=bool)
    # 2. 构造目标时间点的累计病例与动态对照，提前删失者不强制记为阴性
    cases = (event == 1) & (time <= eval_time_point)
    controls = time > eval_time_point
    use = cases | controls
    # 3. 训练集删失分布产生病例IPCW权重；对照组权重在同一时间点为常数，ROC中可抵消
    y_eval = np.array(list(zip(event, time)), dtype=[("cens", bool), ("time", float)])
    case_weights = censor_model.predict_ipcw(y_eval)
    weights = np.where(cases[use], case_weights[use], 1.0)
    y_true = cases[use].astype(int)
    if len(np.unique(y_true)) != 2:
        print(f"{label_time or eval_time_point}: 目标时点病例或对照不足，跳过ROC")
        return
    fpr, tpr, _ = roc_curve(y_true, risk_score[use], sample_weight=weights)
    auc_val = auc(fpr, tpr)
    # 4. 绘图
    time_label = label_time if label_time else f"{eval_time_point}-day"
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"Time-dependent ROC Curve at {time_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/ROC_{model_type.lower().replace(' ', '_')}_{model.__class__.__name__}_validation.png", dpi=300)
    plt.show()
    return auc_val

# 6.1. Cox模型
auc_roc_cox_valid = plot_survival_roc_by_model(model=model_cox, pred_data=val_data_cox,
    time=val_data_cox["time"].values, cens=val_data_cox["cens"].values,
    eval_time_point=validation_time_point, label_time="3 year", model_type="Cox")
# 原来的普通二分类ROC AUC与动态AUC不一致，并不能直接解释为模型效果差
# 6.2. 正则化Cox模型
auc_roc_penaltycox_valid = plot_survival_roc_by_model(model=penaltycox_model, pred_data=X_val,
    time=y_val["time"], cens=y_val["cens"].astype(int),
    eval_time_point=validation_time_point, label_time="3 year", model_type="Machine Survival")
# 6.3. 生存决策树模型
auc_roc_survtree_valid = plot_survival_roc_by_model(model=survtree_model, pred_data=X_val,
    time=y_val["time"], cens=y_val["cens"].astype(int),
    eval_time_point=validation_time_point, label_time="3 year", model_type="Machine Survival")
# 6.4. 随机生存森林模型
auc_roc_survrf_valid = plot_survival_roc_by_model(model=survrf_model, pred_data=X_val,
    time=y_val["time"], cens=y_val["cens"].astype(int),
    eval_time_point=validation_time_point, label_time="3 year", model_type="Machine Survival")
# 6.5. 生存梯度提升模型
auc_roc_survgb_valid = plot_survival_roc_by_model(model=survgb_model, pred_data=X_val,
    time=y_val["time"], cens=y_val["cens"].astype(int),
    eval_time_point=validation_time_point, label_time="3 year", model_type="Machine Survival")
# 6.6. 生存支持向量机模型
auc_roc_survsvm_valid = plot_survival_roc_by_model(model=survsvm_model, pred_data=X_val,
    time=y_val["time"], cens=y_val["cens"].astype(int),
    eval_time_point=validation_time_point, label_time="3 year", model_type="Machine Survival")
################################################################################################
################################## 测试数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
## 构建测试数据集 ##
# 构建测试数据集（对于Cox模型）
with open('outputs/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)
significant_vars_multi
test_data_notscaled = pd.read_csv("outputs/data_splits/test_raw.csv", index_col=0)
test_data_cox = test_data_notscaled[significant_vars_multi + ['time', 'cens']].copy()
# 构造测试数据集（对于机器学习生存模型）
test_data_scaled = pd.read_csv("outputs/data_processed/test_scaled.csv", index_col=0)  # 读取标准化后的测试集
X_test = test_data_scaled.drop(columns=["time", "cens"])  # 构造生存对象 y_test 和协变量 X_test
y_test = np.array(
    [(bool(c), t) for c, t in zip(test_data_scaled["cens"], test_data_scaled["time"])],
    dtype=[("cens", "bool"), ("time", "float")])
# 与验证集保持相同的主要评价时间点，不根据测试集结果重新选择
test_time_point = validation_time_point
## run end ##

############################## 1. C-index 及其95%置信区间 ##################################
# 1.1. Cox模型
cindex_cox_test, ci_lower_cox_test, ci_upper_cox_test = \
    bootstrap_cindex_survmodel(model=model_cox, X=test_data_cox, y=None, model_type="Cox")
print(f"Cox模型测试集 C-index: {cindex_cox_test:.4f} (95% CI: [{ci_lower_cox_test:.4f}, {ci_upper_cox_test:.4f}])")
# 1.2. 正则化Cox模型
cindex_penaltycox_test, ci_lower_penaltycox_test, ci_upper_penaltycox_test = \
    bootstrap_cindex_survmodel(model=penaltycox_model, X=X_test, y=y_test, model_type="Machine Survival")
print(f"正则化Cox模型测试集 C-index: {cindex_penaltycox_test:.4f} (95% CI: [{ci_lower_penaltycox_test:.4f}, {ci_upper_penaltycox_test:.4f}])")
# 1.3. 生存决策树模型
cindex_survtree_test, ci_lower_survtree_test, ci_upper_survtree_test = \
    bootstrap_cindex_survmodel(model=survtree_model, X=X_test, y=y_test, model_type="Machine Survival")
print(f"生存决策树模型测试集 C-index: {cindex_survtree_test:.4f} (95% CI: [{ci_lower_survtree_test:.4f}, {ci_upper_survtree_test:.4f}])")
# 1.4. 随机生存森林模型
cindex_survrf_test, ci_lower_survrf_test, ci_upper_survrf_test = \
    bootstrap_cindex_survmodel(model=survrf_model, X=X_test, y=y_test, model_type="Machine Survival")
print(f"随机生存森林模型测试集 C-index: {cindex_survrf_test:.4f} (95% CI: [{ci_lower_survrf_test:.4f}, {ci_upper_survrf_test:.4f}])")
# 1.5. 生存梯度提升模型
cindex_survgb_test, ci_lower_survgb_test, ci_upper_survgb_test = \
    bootstrap_cindex_survmodel(model=survgb_model, X=X_test, y=y_test, model_type="Machine Survival")
print(f"生存梯度提升模型测试集 C-index: {cindex_survgb_test:.4f} (95% CI: [{ci_lower_survgb_test:.4f}, {ci_upper_survgb_test:.4f}])")
# 1.6. 生存支持向量机模型
cindex_survsvm_test, ci_lower_survsvm_test, ci_upper_survsvm_test = \
    bootstrap_cindex_survmodel(model=survsvm_model, X=X_test, y=y_test, model_type="Machine Survival")
print(f"生存支持向量机模型测试集 C-index: {cindex_survsvm_test:.4f} (95% CI: [{ci_lower_survsvm_test:.4f}, {ci_upper_survsvm_test:.4f}])")
# 保存为 CSV 表格
df_cindex_test = pd.DataFrame({
    "模型名称": ["Cox模型", "正则化Cox模型", "生存决策树模型", "随机生存森林模型", "生存梯度提升模型", "生存支持向量机模型"],
    "C-index": [cindex_cox_test, cindex_penaltycox_test, cindex_survtree_test, cindex_survrf_test, cindex_survgb_test, cindex_survsvm_test],
    "95%CI下限": [ci_lower_cox_test, ci_lower_penaltycox_test, ci_lower_survtree_test, ci_lower_survrf_test, ci_lower_survgb_test, ci_lower_survsvm_test],
    "95%CI上限": [ci_upper_cox_test, ci_upper_penaltycox_test, ci_upper_survtree_test, ci_upper_survrf_test, ci_upper_survgb_test, ci_upper_survsvm_test]
})
df_cindex_test.to_csv("outputs/tables/model_cindex_test.csv", index=False, encoding="utf-8-sig")

###################### 2. 时间依赖性AUC（Time-dependent AUC）及绘制其折线图 ##########################
# 观察测试集的实际随访时间分布
plt.figure(figsize=(8, 5))
plt.hist(test_data_cox['time'], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Time (days)")
plt.ylabel("Frequency")
plt.title("Histogram of Observed Follow-up Time (Test Set)")
plt.tight_layout()
plt.savefig("outputs/figures/evaluation/observed_time_histogram_test.png", dpi=300)
plt.show()
plt.close()
# 绘制测试集时间依赖性AUC折线图的函数（复用验证集的计算方法，不覆盖其图片）
def plot_time_dependent_auc_test(model, X, y_struct, eval_times, label="Model", model_type="Cox"):
    # 较高预测得分代表更高事件风险，使用训练集估计删失分布
    pred_score = get_risk_score(model, X, model_type=model_type)
    auc_vals, mean_auc = cumulative_dynamic_auc(y_train, y_struct, pred_score, times=eval_times)
    print(f"{label} 测试集平均时间依赖AUC: {mean_auc:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(eval_times / (365 / 12), auc_vals, marker='o', linewidth=1, label=f"{label}")
    plt.xlabel("Time (months)")
    plt.ylabel("Time-dependent AUC")
    plt.title("Time-dependent AUC Curve (Test Set)")
    plt.ylim(0., 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/auc_{label.lower().replace(' ', '_')}_test.png", dpi=300)
    plt.show()
    plt.close()
    return(mean_auc, auc_vals)

# 评估时间点与验证集一致（6个月至5年，不使用原代码的5.5年和6年）
eval_times = np.array([180, 270, 365, 365*1.5, 365*2, 365*2.5, 365*3, 365*3.5,
                       365*4, 365*4.5, 365*5])
# 2.1 Cox模型
mean_auc_cox_test, auc_vals_cox_test = plot_time_dependent_auc_test(model=model_cox, X=test_data_cox,
    y_struct=y_test, eval_times=eval_times, label="Cox Model", model_type="Cox")
# 2.2 正则化Cox模型
mean_auc_penaltycox_test, auc_vals_penaltycox_test = plot_time_dependent_auc_test(model=penaltycox_model, X=X_test,
    y_struct=y_test, eval_times=eval_times, label="Regularized Cox Model", model_type="Machine Survival")
# 2.3 生存决策树模型
mean_auc_survtree_test, auc_vals_survtree_test = plot_time_dependent_auc_test(model=survtree_model, X=X_test,
    y_struct=y_test, eval_times=eval_times, label="Survival Tree", model_type="Machine Survival")
# 2.4 随机生存森林模型
mean_auc_survrf_test, auc_vals_survrf_test = plot_time_dependent_auc_test(model=survrf_model, X=X_test,
    y_struct=y_test, eval_times=eval_times, label="Random Survival Forest", model_type="Machine Survival")
# 2.5 生存梯度提升模型
mean_auc_survgb_test, auc_vals_survgb_test = plot_time_dependent_auc_test(model=survgb_model, X=X_test,
    y_struct=y_test, eval_times=eval_times, label="Gradient Boosting Survival", model_type="Machine Survival")
# 2.6 生存支持向量机模型
mean_auc_survsvm_test, auc_vals_survsvm_test = plot_time_dependent_auc_test(model=survsvm_model, X=X_test,
    y_struct=y_test, eval_times=eval_times, label="Survival SVM", model_type="Machine Survival")
# 多模型测试集时间依赖AUC对比折线图 #
plt.figure(figsize=(10, 6))
plt.plot(eval_times / (365 / 12), auc_vals_cox_test, marker='o', label="Cox Model")
plt.plot(eval_times / (365 / 12), auc_vals_penaltycox_test, marker='o', label="Regularized Cox Model")
plt.plot(eval_times / (365 / 12), auc_vals_survtree_test, marker='o', label="Survival Tree")
plt.plot(eval_times / (365 / 12), auc_vals_survrf_test, marker='o', label="Random Survival Forest")
plt.plot(eval_times / (365 / 12), auc_vals_survgb_test, marker='o', label="Gradient Boosting Survival")
plt.plot(eval_times / (365 / 12), auc_vals_survsvm_test, marker='o', label="Survival SVM")
plt.xlabel("Time (months)")
plt.ylabel("Time-dependent AUC")
plt.title("Time-dependent AUC Comparison (Test Set)")
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/evaluation/AUC_curves_allmodel_test.png", dpi=300)
plt.show()
plt.close()
# 时间依赖AUC估计值整理保存为CSV表格（测试集）#
df_auc_test = pd.DataFrame({
    "评估时间（月）": np.insert(np.round(eval_times / (365 / 12), 2).astype(str), 0, "平均AUC"),
    "Cox模型": np.insert(auc_vals_cox_test, 0, mean_auc_cox_test),
    "正则化Cox模型": np.insert(auc_vals_penaltycox_test, 0, mean_auc_penaltycox_test),
    "生存决策树模型": np.insert(auc_vals_survtree_test, 0, mean_auc_survtree_test),
    "随机生存森林模型": np.insert(auc_vals_survrf_test, 0, mean_auc_survrf_test),
    "生存梯度提升模型": np.insert(auc_vals_survgb_test, 0, mean_auc_survgb_test),
    "生存支持向量机模型": np.insert(auc_vals_survsvm_test, 0, mean_auc_survsvm_test)
})
df_auc_test.to_csv("outputs/tables/model_time_auc_test.csv", index=False, encoding="utf-8-sig")

#################################### 3. 绘制某个指定时间点的校准曲线 ####################################
# 绘制测试集校准点（预测生存概率与KM观察生存概率）
def plot_calibration_points_test(pred_surv_probs, time, event, time_point, n_groups, label):
    df_calib = pd.DataFrame({"pred_surv": pred_surv_probs, "time": time, "event": event})
    df_calib["group"] = pd.qcut(df_calib["pred_surv"], n_groups, labels=False, duplicates='drop')
    observed_surv, predicted_surv = [], []
    kmf = KaplanMeierFitter()
    for i in range(df_calib["group"].nunique()):
        group_data = df_calib[df_calib["group"] == i]
        kmf.fit(durations=group_data["time"], event_observed=group_data["event"])
        if group_data["time"].max() < time_point and kmf.survival_function_.iloc[-1, 0] > 0:
            print(f"{label}: 分组{i}在目标时间之前结束观察，跳过该组校准点")
            continue
        observed_surv.append(float(kmf.predict(time_point)))
        predicted_surv.append(group_data["pred_surv"].mean())
    if len(observed_surv) < 2:
        print(f"{label}: 有效校准分组不足，跳过校准曲线")
        return
    plt.figure(figsize=(8, 6))
    plt.plot(predicted_surv, observed_surv, marker='o', linestyle='-', label=f"{label} Calibration")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    plt.xlabel("Predicted Survival Probability")
    plt.ylabel("Observed Survival Probability (KM)")
    plt.title(f"Calibration Curve at {time_point / 365:g} Years (Test Set)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/calibration_{label.lower().replace(' ', '_')}_test.png", dpi=300)
    plt.show()
    plt.close()

# 绘制校准曲线的函数（适用于Cox模型）
def plot_calibration_curve_cox_test(model, data, time_col="time", event_col="cens",
                                    label="Cox Model", time_point=365*3, n_groups=3):
    surv_func_df = model.predict_survival_function(data.drop(columns=[time_col, event_col]), times=[time_point])
    pred_surv_probs = surv_func_df.iloc[0, :].to_numpy()
    plot_calibration_points_test(pred_surv_probs, data[time_col].to_numpy(), data[event_col].to_numpy(),
                                 time_point, n_groups, label)

# 绘制校准曲线的函数（适用于机器学习模型）#
def plot_calibration_curve_ml_test(model, X, y, time_point=365*3, n_groups=3, label="ML Survival Model"):
    if not hasattr(model, "predict_survival_function"):
        print(f"模型 {label} 不支持 survival function 输出，跳过校准曲线绘制。")
        return
    surv_funcs = model.predict_survival_function(X)
    pred_surv_probs = np.array([fn(time_point) for fn in surv_funcs])
    plot_calibration_points_test(pred_surv_probs, y["time"], y["cens"], time_point, n_groups, label)

# 3.1. Cox模型（测试集）
plot_calibration_curve_cox_test(model=model_cox, data=test_data_cox,
                                n_groups=3, label="Cox Model", time_point=test_time_point)
# 3.2. 正则化Cox模型（测试集）
plot_calibration_curve_ml_test(model=penaltycox_model, X=X_test, y=y_test,
                               n_groups=3, label="Regularized Cox Model", time_point=test_time_point)
# 3.3. 生存决策树模型（测试集）
plot_calibration_curve_ml_test(model=survtree_model, X=X_test, y=y_test,
                               n_groups=3, label="Survival Tree Model", time_point=test_time_point)
# 3.4. 随机生存森林模型（测试集）
plot_calibration_curve_ml_test(model=survrf_model, X=X_test, y=y_test,
                               n_groups=3, label="Random Survival Forest", time_point=test_time_point)
# 3.5. 生存梯度提升模型（测试集）
plot_calibration_curve_ml_test(model=survgb_model, X=X_test, y=y_test,
                               n_groups=3, label="Gradient Boosting Survival", time_point=test_time_point)
# 3.6. 生存支持向量机模型（不支持生存概率）
plot_calibration_curve_ml_test(model=survsvm_model, X=X_test, y=y_test,
                               n_groups=3, label="Survival SVM", time_point=test_time_point)

###################### 4. 绘制决策分析曲线 ##########################
# 复用验证集阶段的IPCW净收益计算，单独保存测试集的图片
def plot_dca_curve_test(thresholds, net_benefit_model, net_benefit_all, net_benefit_none, label="Model"):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model, label=label, color="blue", linewidth=2)
    plt.plot(thresholds, net_benefit_all, label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefit_none, label="Treat None", color="green", linestyle="--", linewidth=2)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.ylim(-0.15, np.nanmax(np.asarray(net_benefit_model)) + 0.1)
    plt.title("Decision Curve Analysis (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/dca_{label.lower().replace(' ', '_')}_test.png", dpi=300)
    plt.show()
    plt.close()

# 定义净收益计算和DCA绘图函数（测试集）#
def plot_dca_survival_test(model, X, time, event, time_point=365*3,
                           label="Model", model_type="Cox", thresholds=np.linspace(0.01, 0.99, 100)):
    # 1. 获取预测的生存概率
    if model_type == "Cox":
        surv_funcs = model.predict_survival_function(X.drop(columns=["time", "cens"]), times=[time_point])
        pred_surv_probs = surv_funcs.iloc[0, :].to_numpy()
    elif model_type == "ML Survival":
        surv_funcs = model.predict_survival_function(X)
        pred_surv_probs = np.array([fn(time_point) for fn in surv_funcs])
    else:
        raise ValueError("model_type must be either 'Cox' or 'ML Survival'")
    # 2. 使用训练集删失分布计算测试集的净收益
    net_benefit_model, net_benefit_all, net_benefit_none = calculate_net_benefit_survival(
        time=time, event=event, pred_surv_probs=pred_surv_probs, time_point=time_point, thresholds=thresholds)
    plot_dca_curve_test(thresholds, net_benefit_model, net_benefit_all, net_benefit_none, label=label)

# 4.1. Cox模型
plot_dca_survival_test(model=model_cox, X=test_data_cox, time=test_data_cox["time"].values,
                       event=test_data_cox["cens"].values, time_point=test_time_point,
                       label="Cox Model", model_type="Cox")
# 4.2. 正则化Cox模型
plot_dca_survival_test(model=penaltycox_model, X=X_test, time=y_test["time"],
                       event=y_test["cens"].astype(int), time_point=test_time_point,
                       label="Regularized Cox", model_type="ML Survival")
# 4.3. 生存决策树模型
plot_dca_survival_test(model=survtree_model, X=X_test, time=y_test["time"],
                       event=y_test["cens"].astype(int), time_point=test_time_point,
                       label="Survival Tree", model_type="ML Survival")
# 4.4. 随机生存森林模型
plot_dca_survival_test(model=survrf_model, X=X_test, time=y_test["time"],
                       event=y_test["cens"].astype(int), time_point=test_time_point,
                       label="Random Survival Forest", model_type="ML Survival")
# 4.5. 生存梯度提升模型
plot_dca_survival_test(model=survgb_model, X=X_test, time=y_test["time"],
                       event=y_test["cens"].astype(int), time_point=test_time_point,
                       label="Gradient Boosting Survival", model_type="ML Survival")
# 生存SVM不能直接输出生存概率，不参与DCA

# 各模型3年DCA曲线对比汇总图（不含SVM）#
time_point = test_time_point
thresholds = np.linspace(0.01, 0.99, 100)
# 计算各模型3年预测生存概率
surv_cox = model_cox.predict_survival_function(test_data_cox.drop(columns=["time", "cens"]), times=[time_point])
p_cox = surv_cox.iloc[0, :].to_numpy()
p_penaltycox = np.array([fn(time_point) for fn in penaltycox_model.predict_survival_function(X_test)])
p_tree = np.array([fn(time_point) for fn in survtree_model.predict_survival_function(X_test)])
p_rsf = np.array([fn(time_point) for fn in survrf_model.predict_survival_function(X_test)])
p_gbs = np.array([fn(time_point) for fn in survgb_model.predict_survival_function(X_test)])
# 计算各模型净收益（全部使用同一测试集及训练集删失权重）
nb_cox, nb_all, nb_none = calculate_net_benefit_survival(y_test["time"], y_test["cens"], p_cox, time_point, thresholds)
nb_penaltycox, _, _ = calculate_net_benefit_survival(y_test["time"], y_test["cens"], p_penaltycox, time_point, thresholds)
nb_tree, _, _ = calculate_net_benefit_survival(y_test["time"], y_test["cens"], p_tree, time_point, thresholds)
nb_rsf, _, _ = calculate_net_benefit_survival(y_test["time"], y_test["cens"], p_rsf, time_point, thresholds)
nb_gbs, _, _ = calculate_net_benefit_survival(y_test["time"], y_test["cens"], p_gbs, time_point, thresholds)
# 绘制汇总图
plt.figure(figsize=(10, 6))
plt.plot(thresholds, nb_cox, label="Cox Model", linewidth=2)
plt.plot(thresholds, nb_penaltycox, label="Regularized Cox", linewidth=2)
plt.plot(thresholds, nb_tree, label="Survival Tree", linewidth=2)
plt.plot(thresholds, nb_rsf, label="Random Survival Forest", linewidth=2)
plt.plot(thresholds, nb_gbs, label="Gradient Boosting Survival", linewidth=2)
plt.plot(thresholds, nb_all, label="Treat All", color="red", linestyle="--")
plt.plot(thresholds, nb_none, label="Treat None", color="green", linestyle="--")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (3-Year, Test Set)")
plt.ylim(-0.15, np.nanmax([nb_cox, nb_penaltycox, nb_tree, nb_rsf, nb_gbs]) + 0.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/figures/evaluation/DCA_curves_allmodel_test.png", dpi=300)
plt.show()
plt.close()
# 各模型3年DCA净收益汇总（仅用于当前阈值范围的教学描述）#
summary = []
summary.append(summarize_dca(nb_all, thresholds))
summary.append(summarize_dca(nb_cox, thresholds))
summary.append(summarize_dca(nb_penaltycox, thresholds))
summary.append(summarize_dca(nb_tree, thresholds))
summary.append(summarize_dca(nb_rsf, thresholds))
summary.append(summarize_dca(nb_gbs, thresholds))
df_dca_test = pd.DataFrame(summary, columns=["最大净收益", "对应阈值", "平均净收益"])
df_dca_test.insert(0, "模型名称", ["治疗所有人", "Cox模型", "正则化Cox模型", "生存决策树模型", "随机生存森林模型", "生存梯度提升模型"])
df_dca_test.to_csv("outputs/tables/model_dca_3year_test.csv", index=False, encoding="utf-8-sig")

###################### 5. 基于模型预测值分为低、中、高风险组，绘制KM生存曲线 ##########################
# 绘制测试集KM曲线，使用训练集确定的风险分组界值
def plot_km_by_model_prediction_test(model, pred_data, time, event,
                                     label="Model", model_type="Cox",
                                     time_point=365*3, n_groups=3, reference_data=None):
    """
    根据模型预测风险分组绘制测试集Kaplan-Meier曲线及整体Log-rank检验
    分组阈值由训练集确定，避免在测试集重新划分风险界值
    """
    if reference_data is None:
        reference_data = train_data_cox if model_type == "Cox" else X_train
    # 1. 获取风险得分（越大代表风险越高）
    if model_type == "Cox":
        pred_score = get_risk_score(model, pred_data, model_type="Cox")
        train_score = get_risk_score(model, reference_data, model_type="Cox")
    elif model_type == "ML Survival":
        surv_funcs = model.predict_survival_function(pred_data)
        train_funcs = model.predict_survival_function(reference_data)
        pred_score = 1 - np.array([fn(time_point) for fn in surv_funcs])
        train_score = 1 - np.array([fn(time_point) for fn in train_funcs])
    elif model_type == "SVM":
        pred_score = get_risk_score(model, pred_data, model_type="SVM")
        train_score = get_risk_score(model, reference_data, model_type="SVM")
    else:
        raise ValueError("model_type must be 'Cox', 'ML Survival' or 'SVM'")
    # 2. 使用训练集分位数阈值对测试集分组
    cuts = np.unique(np.quantile(train_score, np.arange(1, n_groups) / n_groups))
    group_labels = ["Low", "Medium", "High"] if len(cuts) == 2 else [f"Group {i+1}" for i in range(len(cuts)+1)]
    df_km = pd.DataFrame({"time": np.asarray(time), "event": np.asarray(event), "pred_score": pred_score})
    df_km["risk_group"] = pd.cut(df_km["pred_score"], bins=[-np.inf, *cuts, np.inf], labels=group_labels)
    # 3. 绘制KM曲线
    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    present_groups = []
    for group in group_labels:
        group_data = df_km[df_km["risk_group"] == group]
        if len(group_data) == 0:
            continue
        present_groups.append(group)
        kmf.fit(durations=group_data["time"], event_observed=group_data["event"], label=group)
        kmf.plot_survival_function(ci_show=False, linewidth=2)
    plt.title(f"Kaplan-Meier Survival Curve ({label}, Test Set)")
    plt.xlabel("Time (days)")
    plt.ylim(0, 1.02)
    plt.ylabel("Event-free Survival Probability")
    plt.legend(title="Risk Group")
    plt.grid(True)
    # 4. Log-rank检验（用于风险分层描述）
    if len(present_groups) > 1:
        results = multivariate_logrank_test(df_km["time"], df_km["risk_group"], event_observed=df_km["event"])
        p_value = results.p_value
        p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
        plt.text(0.05, 0.1, f"Log-rank P {p_str}", transform=plt.gca().transAxes,
                 horizontalalignment='left', fontsize=13,
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/KM_{label.lower().replace(' ', '_')}_test.png", dpi=300)
    plt.show()
    plt.close()

# 5.1. Cox模型
plot_km_by_model_prediction_test(model=model_cox, pred_data=test_data_cox,
                                 time=test_data_cox["time"].values, event=test_data_cox["cens"].values,
                                 label="Cox Model", model_type="Cox", reference_data=train_data_cox)
# 5.2. 正则化Cox模型
plot_km_by_model_prediction_test(model=penaltycox_model, pred_data=X_test,
                                 time=y_test["time"], event=y_test["cens"].astype(int),
                                 label="Regularized Cox Model", model_type="ML Survival",
                                 time_point=test_time_point, reference_data=X_train)
# 5.3. 生存决策树模型
plot_km_by_model_prediction_test(model=survtree_model, pred_data=X_test,
                                 time=y_test["time"], event=y_test["cens"].astype(int),
                                 label="Survival Tree", model_type="ML Survival",
                                 time_point=test_time_point, reference_data=X_train)
# 5.4. 随机生存森林模型
plot_km_by_model_prediction_test(model=survrf_model, pred_data=X_test,
                                 time=y_test["time"], event=y_test["cens"].astype(int),
                                 label="Random Survival Forest", model_type="ML Survival",
                                 time_point=test_time_point, reference_data=X_train)
# 5.5. 生存梯度提升模型
plot_km_by_model_prediction_test(model=survgb_model, pred_data=X_test,
                                 time=y_test["time"], event=y_test["cens"].astype(int),
                                 label="Gradient Boosting Survival", model_type="ML Survival",
                                 time_point=test_time_point, reference_data=X_train)
# 5.6. 生存支持向量机模型
plot_km_by_model_prediction_test(model=survsvm_model, pred_data=X_test,
                                 time=y_test["time"], event=y_test["cens"].astype(int),
                                 label="Survival SVM", model_type="SVM", reference_data=X_train)

###################### 6. 绘制指定时间点的ROC曲线（IPCW校正右删失） ##########################
# 绘制测试集三年动态ROC，与验证集使用相同病例/对照及删失权重定义
def plot_survival_roc_by_model_test(model, pred_data, time, cens,
                                    eval_time_point=365*3, label_time=None, model_type="Cox"):
    # 1. 获取风险得分
    risk_score = get_risk_score(model, pred_data, model_type=model_type)
    time, event = np.asarray(time, dtype=float), np.asarray(cens, dtype=bool)
    # 2. 目标时间点之前观察到的事件为病例，超过目标时间点仍在观察为对照
    cases = event & (time <= eval_time_point)
    controls = time > eval_time_point
    use = cases | controls
    # 3. 使用训练集的删失分布估计病例权重
    y_eval = np.array(list(zip(event, time)), dtype=[("cens", bool), ("time", float)])
    case_weights = censor_model.predict_ipcw(y_eval)
    weights = np.where(cases[use], case_weights[use], 1.0)
    y_true = cases[use].astype(int)
    if len(np.unique(y_true)) != 2:
        print(f"{label_time or eval_time_point}: 目标时点病例或对照不足，跳过ROC")
        return
    fpr, tpr, _ = roc_curve(y_true, risk_score[use], sample_weight=weights)
    auc_val = auc(fpr, tpr)
    # 4. 绘图
    time_label = label_time if label_time else f"{eval_time_point}-day"
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"Time-dependent ROC Curve at {time_label} (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/evaluation/ROC_{model_type.lower().replace(' ', '_')}_{model.__class__.__name__}_test.png", dpi=300)
    plt.show()
    plt.close()
    return auc_val

# 6.1. Cox模型
auc_roc_cox_test = plot_survival_roc_by_model_test(model=model_cox, pred_data=test_data_cox,
    time=test_data_cox["time"].values, cens=test_data_cox["cens"].values,
    eval_time_point=test_time_point, label_time="3 year", model_type="Cox")
# 6.2. 正则化Cox模型
auc_roc_penaltycox_test = plot_survival_roc_by_model_test(model=penaltycox_model, pred_data=X_test,
    time=y_test["time"], cens=y_test["cens"].astype(int),
    eval_time_point=test_time_point, label_time="3 year", model_type="Machine Survival")
# 6.3. 生存决策树模型
auc_roc_survtree_test = plot_survival_roc_by_model_test(model=survtree_model, pred_data=X_test,
    time=y_test["time"], cens=y_test["cens"].astype(int),
    eval_time_point=test_time_point, label_time="3 year", model_type="Machine Survival")
# 6.4. 随机生存森林模型
auc_roc_survrf_test = plot_survival_roc_by_model_test(model=survrf_model, pred_data=X_test,
    time=y_test["time"], cens=y_test["cens"].astype(int),
    eval_time_point=test_time_point, label_time="3 year", model_type="Machine Survival")
# 6.5. 生存梯度提升模型
auc_roc_survgb_test = plot_survival_roc_by_model_test(model=survgb_model, pred_data=X_test,
    time=y_test["time"], cens=y_test["cens"].astype(int),
    eval_time_point=test_time_point, label_time="3 year", model_type="Machine Survival")
# 6.6. 生存支持向量机模型
auc_roc_survsvm_test = plot_survival_roc_by_model_test(model=survsvm_model, pred_data=X_test,
    time=y_test["time"], cens=y_test["cens"].astype(int),
    eval_time_point=test_time_point, label_time="3 year", model_type="Machine Survival")

################################################################################################
################################## 模型的SHAP解释 ###############################################
################################################################################################
## 编写绘制可解释性图片的函数，方便调用 ##
def shap_explain_survival_model(model, X_train, X_test, model_name="Cox",
                                 model_type="Cox",               # "Cox" or "ML Survival"
                                 pred_target="auto",             # "risk" / "death_prob" / "auto"
                                 time_horizon=365*3,             # 用于3年事件发生概率解释，与前面的主要评价时点一致
                                 n_interpret=50,
                                 obs_index=5,
                                 scatter_x1="tgrade", scatter_x2="progrec", # 可自定义 scatter 图的变量
                                 save_path="outputs/figures/shap/"):
    """
    生存模型 SHAP 可解释性分析（含所有关键图）

    支持：
    - lifelines.CoxPHFitter（解释3年事件发生概率；也支持原来的相对风险得分）
    - sksurv 机器学习模型（解释3年事件发生概率）
    - FastSurvivalSVM（解释风险排序得分，不是3年事件发生概率）
    """
    os.makedirs(save_path, exist_ok=True)
    # 自动识别解释目标：SVM解释风险得分，其余模型解释同一时间点的事件发生概率（1 - S(t)）
    if pred_target == "auto":
        pred_target = "risk" if model_type == "SVM" else "death_prob"
    # 构建预测函数（SHAP传入的可能是ndarray，需要恢复原始变量名和列顺序）
    def model_predict(X):
        X = pd.DataFrame(X, columns=X_train.columns)
        if model_type == "Cox" and pred_target == "risk":
            return model.predict_partial_hazard(X).to_numpy().reshape(-1)
        elif model_type == "Cox" and pred_target == "death_prob":
            surv_df = model.predict_survival_function(X, times=[time_horizon])
            return 1 - surv_df.iloc[0, :].to_numpy()
        elif model_type == "ML Survival" and pred_target == "death_prob":
            surv_fns = model.predict_survival_function(X)
            return 1 - np.array([fn(time_horizon) for fn in surv_fns])
        elif model_type == "SVM" and pred_target == "risk":
            pred_score = model.predict(X)
            return -pred_score if model.rank_ratio < 1.0 else pred_score  # 与前面C-index一致的风险方向
        else:
            raise ValueError(f"暂不支持该组合：model_type={model_type}, pred_target={pred_target}")

    # SHAP 解释器（只从训练集选取背景样本，不使用测试集拟合背景分布）
    np.random.seed(123)
    background_data = shap.sample(X_train, min(30, len(X_train)), random_state=123)
    explainer = shap.Explainer(model_predict, background_data, algorithm="permutation", seed=123)
    shap_values = explainer(X_test.iloc[:n_interpret], max_evals=3 * (2 * X_train.shape[1] + 1), silent=True)

    # 检查 SHAP 贡献之和是否重建对应模型预测值
    pred_values = model_predict(X_test.iloc[:n_interpret])
    reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
    print(f"{model_name} SHAP加和误差：{np.max(np.abs(pred_values - reconstructed)):.2e}")

    # 1. 条形图（全局特征重要性）
    plt.figure(figsize=(15, 12))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.savefig(f"{save_path}bar_{model_name}.png", dpi=200, bbox_inches='tight')
    plt.close()
    # 2. 蜂群图
    plt.figure(figsize=(15, 12))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.savefig(f"{save_path}beeswarm_{model_name}.png", dpi=200, bbox_inches='tight')
    plt.close()
    # 3. 散点图（自定义变量）
    if scatter_x1 in X_test.columns and scatter_x2 in X_test.columns:
        plt.figure(figsize=(15, 12))
        shap.plots.scatter(shap_values[:, scatter_x1], color=shap_values[:, scatter_x2], show=False)
        plt.savefig(f"{save_path}scatter_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
    # 4. 瀑布图（单个观测）
    plt.figure(figsize=(15, 12))
    shap.plots.waterfall(shap_values[obs_index], max_display=15, show=False)
    plt.savefig(f"{save_path}waterfall_{obs_index}_{model_name}.png", dpi=200, bbox_inches='tight')
    plt.close()
    # 5. 力图（单个观测）
    force_plot_sig = shap.plots.force(shap_values[obs_index])
    shap.save_html(f"{save_path}force_plot_sig_{obs_index}_{model_name}.html", force_plot_sig)
    # 6. 力图（所有观测）
    # RSF的背景值可能存在浮点尾差；整体力图需要一个可JSON序列化的标量背景值
    base_value = float(np.mean(shap_values.base_values))
    if not np.allclose(shap_values.base_values, base_value, rtol=0, atol=1e-10):
        raise ValueError("不同样本的SHAP背景值不一致，无法直接绘制整体力图")
    force_plot_all = shap.plots.force(base_value, shap_values.values, features=shap_values.data,
                                      feature_names=shap_values.feature_names)
    shap.save_html(f"{save_path}force_plot_all_{model_name}.html", force_plot_all)

    print(f"[SHAP完成] 模型：{model_name} | 类型：{model_type} | 目标：{pred_target}")
    if pred_target == "death_prob":
        print(f"解释时间点：{time_horizon} 天（事件发生概率）")
    print(f"图像保存目录：{save_path}")
###################### 1. Cox模型的可解释性 ##########################    
shap_explain_survival_model(model=model_cox, X_train=train_data_cox.drop(columns=['time','cens']),
    X_test=test_data_cox.drop(columns=['time','cens']), model_name="Cox", model_type="Cox",
    time_horizon=test_time_point, n_interpret=50, obs_index=5,
    scatter_x1="pnodes", scatter_x2="progrec", save_path="outputs/figures/shap/")
###################### 2. 正则化Cox模型（Coxnet） ##########################    
shap_explain_survival_model(model=penaltycox_model, X_train=X_train,
    X_test=X_test, model_name="Coxnet", model_type="ML Survival",
    time_horizon=test_time_point, n_interpret=50, obs_index=5,
    save_path="outputs/figures/shap/")
###################### 3. 决策树生存模型 ##########################    
shap_explain_survival_model(model=survtree_model, X_train=X_train,
    X_test=X_test, model_name="SurvTree", model_type="ML Survival",
    time_horizon=test_time_point, n_interpret=50, obs_index=5,
    save_path="outputs/figures/shap/")
###################### 4. 随机生存森林模型 ##########################    
shap_explain_survival_model(model=survrf_model, X_train=X_train,
    X_test=X_test, model_name="RSF", model_type="ML Survival",
    time_horizon=test_time_point, n_interpret=50, obs_index=5,
    save_path="outputs/figures/shap/")
###################### 5. 生存梯度提升模型 ##########################    
shap_explain_survival_model(model=survgb_model, X_train=X_train,
    X_test=X_test, model_name="GBSA", model_type="ML Survival",
    time_horizon=test_time_point, n_interpret=50, obs_index=5,
    save_path="outputs/figures/shap/")
###################### 6. 生存支持向量机模型（解释风险得分，不是事件发生概率） ##########################
shap_explain_survival_model(model=survsvm_model, X_train=X_train,
    X_test=X_test, model_name="SurvSVM", model_type="SVM", pred_target="risk",
    n_interpret=50, obs_index=5, save_path="outputs/figures/shap/")
