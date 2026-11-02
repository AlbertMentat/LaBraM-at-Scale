"""排除数据集清单 — 非人类头皮 EEG, 不纳入人类 EEG 预训练/评测.

依据: 真读各数据集原始论文/dataset_description.json 定性 (2026-06-14).
这些用人类 10-20 通道体系不适用, 应从预训练池排除 (不计入可用率分母).
"""

# 动物 EEG/LFP (物种命名体系完全不同于人类 10-20)
ANIMAL_DATASETS = {
    'ds003380',  # 猪 (pig) — Corticothalamic, gradual cerebral ischemia
    'ds004598',  # 大鼠 (TgF344-AD rat) — LFP during linear track
    'ds006269',  # 大鼠 (Syngap1 rats) — Tethered EEG
    'ds006366',  # 小鼠 (Mouse) — Sleep Staging Validation
    'ds007028',  # 猴 (Macaque) — Auditory Cortex DISC
}

# 超高密度/特殊电极阵列 (物理无 10-20 对应, 原始即阵列编号)
NON_1020_ARRAY_DATASETS = {
    'ds007591',  # g.tec g.pangolin 8×16=128 贴语言区超高密度阵列
    'ds007602',  # 同上 (EEG-Speech, g.pangolin)
    'ds007808',  # 同上 (EEG-Speech Brain Decoding)
}

# 非头皮 EEG 信号 (iEEG/眼动/肌电等, 非头皮电极)
NON_SCALP_DATASETS = {
    'ds004588',  # 眼动 (left_x/pupil) — Neuma
    'ds006126',  # 肌电 (FDI_R/ADM_R/APB_R) — TDCS Motor Imagery
    'ds005185',  # 耳道深部 (ELA/ELB) — Ear-EEG (实为耳道, 非头皮10-20)
}

# 通道与 LaBraM standard_1020 匹配率过低 (通道名非标准/特殊电极, 匹配后有效通道≤8%, 信息损失过大)
LOW_CHMATCH_DATASETS = {
    'GrosseWentrup2009',  # 128ch → 匹配0 (通道名全非标准)
    'ds003004',           # BioSemi 224ch → 匹配18 (8%, A1-H32大部分未映射) ImaginedEmotion
    'ds004369',           # → 匹配0 (cEEGrid特殊电极) attendedSpeakerParadigm
    'ds004015',           # 18ch → 匹配0 (cEEGrid耳周电极) AttendedSpeakerParadigmcEEGridAttention
}

# 非标准通道命名 (resolve_channels 后有效通道=0, 2026-06-18 逐一确认首pkl通道名)
# 全部用完整集名(含ON_/ZEN_/figshare_/MOABB_前缀, 因is_excluded按完整名匹配)
NON_STANDARD_CH_DATASETS = {
    # 非EEG传感器
    'ON_ds005873',          # 1ch ECGEMG ACC/GYR 加速度计陀螺仪 (5764h, 非脑电!)
    'figshare_22192831',    # 1ch ECG 心电
    # 设备元数据/空数据列
    'ON_ds007358',          # 50ch TIME_STAMP/COUNTER/BATTERY Emotiv原始导出
    'ZEN_10803229',         # 64ch 'Empty Event Data'
    'figshare_28682165',    # 1ch E1
    'ZEN_15744486', 'ON_ds005688', 'ZEN_8191877', 'ZEN_7253438', 'ON_ds003838',  # 1-4ch E1/E2/X2/ch1 (EOG/单导)
    # 颅内/特殊电极
    'ON_ds004752',          # 19ch AHL1.. 颅内深部电极(海马)
    'ON_ds005207',          # 21ch R8/R7.. ROI/非标准
    'ON_ds005692',          # 25ch SrC6/SrLB 特殊命名
    'ZEN_834976',           # 96ch eog/thumb 眼动肌肉
    # 无位置信息 (MEG导出EEG00N / 纯数字 / 采集器号 / BioSemi EXG / Channel N)
    'ON_ds003645', 'ON_ds003944', 'ON_ds003947',  # 64-75ch EEG001.. (MEG系统导出, 无10-20位置)
    'ON_ds007605', 'ON_ds004460',                  # 128/160ch BrainVision RDA_N 采集器通道号
    'ON_ds004511', 'ON_ds005131', 'MOABB_GrosseWentrup2009', 'ZEN_1217449',  # 64-139ch 纯数字1/2/3
    'ZEN_7277151', 'ON_ds004745', 'figshare_21303342', 'figshare_20102876',  # 纯数字
    'ZEN_4659714',          # 72ch EXG1.. BioSemi外部电极
    'ZEN_5882500', 'mendeley_gvt64rndnd',  # Channel N
}

# 全部排除集 (非人类头皮标准 EEG)
EXCLUDED_DATASETS = (ANIMAL_DATASETS | NON_1020_ARRAY_DATASETS | NON_SCALP_DATASETS
                    | LOW_CHMATCH_DATASETS | NON_STANDARD_CH_DATASETS)


def is_excluded(dataset_name):
    """判断数据集是否应排除 (匹配完整名 或 剥ON_前缀的 dsXXXXX)."""
    name = str(dataset_name).strip()
    base = name.replace('ON_', '').strip()
    for prefix in EXCLUDED_DATASETS:
        # 完整名精确匹配(NON_STANDARD_CH_DATASETS用完整名如ON_ds005873/ZEN_xxx)
        if name == prefix:
            return True
        # dsXXXXX 前缀匹配(ANIMAL/LOW_CHMATCH等用裸dsXXXXX)
        if base == prefix or base.startswith(prefix + '_'):
            return True
    return False
