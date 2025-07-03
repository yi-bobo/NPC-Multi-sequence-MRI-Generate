import sys
sys.path.append('/data1/weiyibo/NPC-MRI/Code/Pctch_model/Utils')

import os
import random
import pandas as pd

def who_description_from_stage(who_stage):
    if who_stage == 1:
        desc = 'Keratinizing SCC.'
    elif who_stage == 2:
        desc = 'Non-Keratinizing Diff.'
    elif who_stage == 3:
        desc = 'Non-Keratinizing Undiff.'
    else:
        desc = 'Faulty'
    return desc

def T_description_from_stage(T_stage):
    if T_stage == 0:
        desc = 'No tumor, biomarkers suggest disease.'
    elif T_stage == 1:
        desc = 'Tumor in nasopharynx, nasal cavity, or oropharynx.'
    elif T_stage == 2:
        desc = 'Tumor in parapharyngeal space.'
    elif T_stage == 3:
        desc = 'Tumor in bone or sinuses.'
    elif T_stage == 4:
        desc = 'Tumor invading skull, cranial nerves, or beyond.'
    else:
        desc = 'Faulty'
    return desc

def N_description_from_stage(N_stage):
    if N_stage == 0:
        desc = 'No lymph node metastasis.'
    elif N_stage == 1:
        desc = 'Unilateral node ≤6 cm or retropharyngeal node.'
    elif N_stage == 2:
        desc = 'Bilateral nodes ≤6 cm.'
    elif N_stage == 3:
        desc = 'Node >6 cm or in supraclavicular fossa.'
    else:
        desc = 'Faulty'
    return desc

def M_description_from_stage(M_stage):
    if M_stage == 0:
        desc = 'No distant metastasis.'
    elif M_stage == 1:
        desc = 'Distant metastasis present.'
    else:
        desc = 'Faulty'
    return desc

def wich_gender(gender):
    if gender == 1:
        desc = 'Male'
    elif gender == 2:
        desc = 'Female'
    else:
        desc = 'Faulty'
    return desc

def patient_information(patient_id, df):
    birth = df[df['Patient_ID'] == patient_id]['Birth'].values[0]
    age = df[df['Patient_ID'] == patient_id]['Age'].values[0]
    gender = df[df['Patient_ID'] == patient_id]['Gender'].values[0]
    gender_desc = wich_gender(gender)
    diagnosis_time = df[df['Patient_ID'] == patient_id]['Diagnosis_Time'].values[0]  # 诊断时间
    who = df[df['Patient_ID'] == patient_id]['WHO'].values[0]
    who_desc = who_description_from_stage(who)
    if who_desc == 'Faulty':
        return 'Faulty'
    t = df[df['Patient_ID'] == patient_id]['T'].values[0]
    t_desc = T_description_from_stage(t)
    if t_desc == 'Faulty':
        return 'Faulty'
    n = df[df['Patient_ID'] == patient_id]['N'].values[0]
    n_desc = N_description_from_stage(n)
    if n_desc == 'Faulty':
        return 'Faulty'
    m = df[df['Patient_ID'] == patient_id]['M'].values[0]
    m_desc = M_description_from_stage(m)
    if m_desc == 'Faulty':
        return 'Faulty'
    infomation = (f"The patient was born in {birth}, and is {age} years old. The gender is {gender_desc}. The diagnosis time is {diagnosis_time}. "
                 f"{who_desc}{t_desc}{n_desc}{m_desc}")
    return infomation

path_dir = "/data1/weiyibo/NPC-MRI/Data/NPCIC/zhongshan2/113_ToNumpy_T1C/"  # path to the directory of the dataset
split_dir = "./Split/zhongshan2"
os.makedirs(split_dir, exist_ok=True)

# 读取文本信息excel文件
patient_info_path = "/data1/weiyibo/NPC-MRI/Data/main_info/main_info/Patient_Info_ZhongShan2.csv"
df = pd.read_csv(patient_info_path)

# 选择包含特定列的行，并去除这些列中包含缺失值的行
required_columns = ['Patient_ID', 'Birth', 'Age', 'Gender', 'WHO', 'T', 'N', 'M']
filtered_df = df[required_columns].dropna(subset=required_columns)

# 提取筛选后的 Patient_ID
df_patient_ids = filtered_df['Patient_ID']

all_patient_ids = []
for patient_id in os.listdir(path_dir):
    if patient_id in df_patient_ids.values:
        all_patient_ids.append(patient_id)

# 划分训练集、验证集、测试集
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
train_num = int(len(all_patient_ids) * train_ratio)
val_num = int(len(all_patient_ids) * val_ratio)
test_num = len(all_patient_ids) - train_num - val_num

random.shuffle(all_patient_ids)
train_patient_ids = all_patient_ids[:train_num]
val_patient_ids = all_patient_ids[train_num:train_num+val_num]
test_patient_ids = all_patient_ids[train_num+val_num:]

# 生成对应文本信息
for train_patient_id in train_patient_ids:
    information = patient_information(train_patient_id, df)
    if information == 'Faulty':
        continue
    patient_path = os.path.join(path_dir, train_patient_id)
    with open(os.path.join(split_dir, f"train_with_info.txt"), "a") as f:
        f.write(f"{patient_path} & {information}\n")

for val_patient_id in val_patient_ids:
    information = patient_information(val_patient_id, df)
    if information == 'Faulty':
        continue
    patient_path = os.path.join(path_dir, val_patient_id)
    with open(os.path.join(split_dir, f"val_with_info.txt"), "a") as f:
        f.write(f"{patient_path} & {information}\n")

for test_patient_id in test_patient_ids:
    information = patient_information(test_patient_id, df)
    if information == 'Faulty':
        continue
    patient_path = os.path.join(path_dir, test_patient_id)
    with open(os.path.join(split_dir, f"test_with_info.txt"), "a") as f:
        f.write(f"{patient_path} & {information}\n")