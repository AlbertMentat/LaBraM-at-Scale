"""通道名映射表 + 归一化规则 — 把各种原始 EEG 命名统一到 LaBraM standard_1020.

覆盖:
  - TUH 'EEG FP1-REF' -> 'FP1' (去前缀/参考后缀)
  - 带采集前缀 'EEGstream EE225_Fp1' -> 'FP1', 'BrainVision RDA_G01' (无标准名,保留检测)
  - EGI GSN-HydroCel 'E22' -> 'FP1' (位置型映射, 按官方网格对应)
"""

# EGI GSN-HydroCel 128/129 网 → 10-20 标准位置 (业界公认对应, Luu&Ferree 2005 / EGI官方)
# key='E<编号>'(大写), value=标准10-20名. 仅列能对应到标准10-20/10-10的电极.
EGI_128_TO_1020 = {
    'E1': 'AF8',
    'E10': 'AF2',
    'E100': 'TP10',
    'E101': 'TP8',
    'E102': 'CCP6',
    'E103': 'C4',
    'E107': 'A2',
    'E108': 'T4',
    'E109': 'C6',
    'E110': 'FC4',
    'E112': 'FC2',
    'E114': 'FT10',
    'E115': 'FT8',
    'E116': 'FC6',
    'E118': 'F2',
    'E122': 'F8',
    'E123': 'F6',
    'E124': 'F4',
    'E125': 'F10',
    'E128': 'F9',
    'E13': 'FC1',
    'E15': 'FPZ',
    'E16': 'AFZ',
    'E18': 'AF1',
    'E2': 'AF6',
    'E20': 'F1',
    'E23': 'AF3',
    'E24': 'F3',
    'E25': 'FP1',
    'E26': 'AF5',
    'E27': 'F5',
    'E3': 'AF4',
    'E32': 'AF7',
    'E33': 'F7',
    'E34': 'FC5',
    'E35': 'FC3',
    'E36': 'C3',
    'E37': 'C1',
    'E39': 'FT7',
    'E40': 'C5',
    'E42': 'CCP3',
    'E44': 'FT9',
    'E45': 'T3',
    'E46': 'CCP5',
    'E5': 'FZ',
    'E50': 'TP7',
    'E51': 'CP5',
    'E52': 'CP3',
    'E53': 'CP1',
    'E54': 'CCP1',
    'E55': 'CZ',
    'E56': 'A1',
    'E57': 'TP9',
    'E58': 'P7',
    'E59': 'P5',
    'E6': 'FCZ',
    'E60': 'P3',
    'E62': 'PZ',
    'E64': 'P9',
    'E65': 'PO7',
    'E66': 'PO3',
    'E67': 'P1',
    'E70': 'O1',
    'E71': 'PO1',
    'E72': 'POZ',
    'E75': 'OZ',
    'E76': 'PO2',
    'E79': 'CCP2',
    'E8': 'FP2',
    'E83': 'O2',
    'E84': 'PO4',
    'E85': 'P4',
    'E86': 'CP2',
    'E87': 'C2',
    'E90': 'PO8',
    'E91': 'P6',
    'E92': 'CP4',
    'E93': 'CCP4',
    'E95': 'P10',
    'E96': 'P8',
    'E97': 'CP6',
}

# EGI 256/257 网 → 10-20 (HydroCel GSN 256)
EGI_256_TO_1020 = {
    'E10': 'AF8',
    'E100': 'P1',
    'E101': 'PZ',
    'E105': 'P9',
    'E106': 'T5',
    'E108': 'PO5',
    'E109': 'PO3',
    'E11': 'AF6',
    'E118': 'PO1',
    'E12': 'AF4',
    'E127': 'PO2',
    'E13': 'AF2',
    'E130': 'CP2',
    'E138': 'OZ',
    'E140': 'PO4',
    'E143': 'CCP2',
    'E144': 'C2',
    'E15': 'FZ',
    'E150': 'O2',
    'E151': 'PO6',
    'E152': 'P4',
    'E154': 'CP4',
    'E161': 'P6',
    'E162': 'CP6',
    'E163': 'CCP4',
    'E164': 'C4',
    'E169': 'T6',
    'E17': 'FC1',
    'E170': 'P8',
    'E172': 'CCP6',
    'E177': 'P10',
    'E179': 'TP8',
    'E18': 'FP2',
    'E181': 'C6',
    'E190': 'TP10',
    'E192': 'T4',
    'E196': 'FC4',
    'E197': 'FC2',
    'E202': 'T8',
    'E204': 'FC6',
    'E209': 'A2',
    'E21': 'AFZ',
    'E211': 'FT8',
    'E219': 'FT10',
    'E221': 'F8',
    'E222': 'F6',
    'E224': 'F4',
    'E226': 'F10',
    'E23': 'F1',
    'E230': 'AF10',
    'E244': 'AF9',
    'E252': 'F9',
    'E26': 'FPZ',
    'E28': 'AF1',
    'E34': 'AF3',
    'E36': 'F3',
    'E37': 'FP1',
    'E38': 'AF5',
    'E46': 'AF7',
    'E48': 'F5',
    'E51': 'FC3',
    'E53': 'C1',
    'E55': 'F7',
    'E57': 'FC5',
    'E6': 'F2',
    'E62': 'FT7',
    'E66': 'C3',
    'E67': 'FT9',
    'E69': 'T7',
    'E71': 'C5',
    'E74': 'T3',
    'E76': 'CCP5',
    'E77': 'CCP3',
    'E78': 'CP3',
    'E79': 'CCP1',
    'E8': 'FCZ',
    'E81': 'CZ',
    'E84': 'TP7',
    'E86': 'CP5',
    'E89': 'CP1',
    'E90': 'CPZ',
    'E92': 'A1',
    'E94': 'TP9',
    'E96': 'P7',
    'E97': 'P5',
    'E99': 'P3',
}

# 带采集软件前缀的剥离模式 (前缀_真名 / 前缀 真名)
_STRIP_PREFIXES = (
    'EEGSTREAM EE225_', 'EEGSTREAM ', 'BRAINVISION RDA_', 'BRAINVISION ',
    'PSG_', 'EEG_',
)

# 双极/参考后缀 (整体去除)
_STRIP_SUFFIXES = (
    '-REF', '-LE', '-RE', '-A1A2', '-A1-A2', '-A1', '-A2',
    '-M1', '-M2', '-AVG', '-AFZ', '-CZ', '-AA', '-AR',
    '_R', '_L', '_A', '_B',
)


def normalize_ch_name(ch):
    """单通道名归一化 (字符串规则, 不含位置型 EGI).

    处理: 采集前缀(EEGstream/PSG_/EEG_) + 'EEG ' 前缀 + 参考/双极后缀
          (-REF/-A1A2/-AFz/_R/_L) + 双极取第一节点(FP1-AFz -> FP1).
    """
    c = str(ch).strip().upper()
    for p in _STRIP_PREFIXES:
        if c.startswith(p):
            c = c[len(p):]
            break
    if c.startswith('EEG '):
        c = c[4:]
    for suf in _STRIP_SUFFIXES:
        if c.endswith(suf):
            c = c[:-len(suf)]
            break
    c = c.strip().lstrip('_').strip()
    # 双极 'FP1-AFZ' / "CZ'-FZ" -> 取第一节点
    if '-' in c:
        head = c.split('-')[0].strip().rstrip("'")
        if head:
            c = head
    return c


def is_egi_names(ch_names):
    """判断是否 EGI E编号命名 (E1..EN, 过半是)."""
    n = sum(1 for c in ch_names
            if str(c).strip().upper().lstrip('E').lstrip('0').isdigit()
            and str(c).strip().upper().startswith('E'))
    return n >= max(5, len(ch_names) * 0.5)


def map_egi_list(ch_names, std_set):
    """EGI E编号列表 → 标准名列表 (位置型). 不能对应的设 None.
    返回 [(idx, std_name)...] 仅保留能映射且在 std_set 的.
    根据电极总数选 128 或 256 网映射表.
    """
    n = len(ch_names)
    table = EGI_256_TO_1020 if n > 200 else EGI_128_TO_1020
    out = []
    for i, c in enumerate(ch_names):
        key = str(c).strip().upper()
        # E01 -> E1 归一(去前导0)
        if key.startswith('E') and key[1:].lstrip('0').isdigit():
            key = 'E' + key[1:].lstrip('0')
        std = table.get(key) or table.get(str(c).strip())
        if std and std in std_set:
            out.append((i, std))
    return out


# Biosemi 64通道 ABC标签 (A1-A32/B1-B32) → 10-20 (mne biosemi64物理顺序)
BIOSEMI64_AB_TO_1020 = {'A1': 'FP1', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7', 'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A2': 'AF7', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9', 'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'IZ', 'A29': 'OZ', 'A3': 'AF3', 'A30': 'POZ', 'A31': 'PZ', 'A32': 'CPZ', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7', 'A9': 'FC5', 'B1': 'FPZ', 'B10': 'F8', 'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCZ', 'B16': 'CZ', 'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B2': 'FP2', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2', 'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B3': 'AF8', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2', 'B4': 'AF4', 'B5': 'AFZ', 'B6': 'FZ', 'B7': 'F2', 'B8': 'F4', 'B9': 'F6'}


def map_biosemi_ab(raw_ch, std_set, n_xch):
    """Biosemi ABC标签解析: '1-A1'/'2-B32' → 标准名 (去被试号前缀N-, 查AB表).
    双人实验保留第一次出现的标准名 (去重)."""
    keep_idx, names, seen = [], [], set()
    for i, c in enumerate(raw_ch):
        if i >= n_xch:
            continue
        s = str(c).strip().upper()
        # 去被试号前缀 '1-' '2-'
        if len(s) > 2 and s[0].isdigit() and s[1] == '-':
            s = s[2:]
        std = BIOSEMI64_AB_TO_1020.get(s)
        if std and std in std_set and std not in seen:
            keep_idx.append(i); names.append(std); seen.add(std)
    return keep_idx, names


def is_biosemi_ab(raw_ch):
    """判断是否 Biosemi ABC标签 (A1-A32/B1-B32, 可带N-前缀)."""
    import re
    n = sum(1 for c in raw_ch if re.match(r'^(\d+-)?[AB]\d{1,2}$', str(c).strip().upper()))
    return n >= max(5, len(raw_ch) * 0.5)


def resolve_channels(raw_ch, std_set, n_xch=None, dataset_id=None):
    """统一通道解析: 原始ch_names列表 → (keep_idx, std_names).

    处理顺序:
      0. 若该数据集有坐标重建映射 (占位通道名集, 从electrodes.tsv几何重建) → 用之
      1. 若整体是 EGI E编号命名 → 用位置映射表 (按总通道数选128/256网)
      2. 否则逐通道字符串归一化 (TUH前缀/采集前缀/参考后缀)
      3. 只保留: 归一化后在 std_set 中 且 索引 < n_xch (修 ch≠X)

    返回:
      keep_idx: 保留的原始通道索引 (用于裁剪 X)
      std_names: 对应的标准通道名 (大写, 在 std_set 中)
    """
    if n_xch is None:
        n_xch = len(raw_ch)
    normalized_ch = [normalize_ch_name(c) for c in raw_ch]
    keep_idx, std_names = [], []
    # 0. per-dataset 坐标重建映射 (占位命名集)
    if dataset_id is not None:
        pdm = get_perdataset_chmap(dataset_id)
        if pdm:
            seen = set()
            for i, c in enumerate(raw_ch):
                if i >= n_xch:
                    continue
                s = pdm.get(str(c).strip().upper())
                if s and s in std_set and s not in seen:
                    keep_idx.append(i); std_names.append(s); seen.add(s)
            if std_names:
                return keep_idx, std_names
    # Biosemi ABC标签 (A1-A32/B1-B32, 可带被试号前缀)
    if is_biosemi_ab(raw_ch):
        ki, names = map_biosemi_ab(raw_ch, std_set, n_xch)
        if names:
            return ki, names
    if is_egi_names(normalized_ch):
        n = len(raw_ch)
        table = EGI_256_TO_1020 if n > 160 else EGI_128_TO_1020
        seen = set()
        for i, key in enumerate(normalized_ch):
            if i >= n_xch:
                continue
            key = key.upper()
            if key.startswith('E') and key[1:].lstrip('0').isdigit():
                key = 'E' + key[1:].lstrip('0')
            s = table.get(key)
            if s and s in std_set and s not in seen:
                keep_idx.append(i); std_names.append(s); seen.add(s)
    else:
        seen = set()
        for i, c in enumerate(raw_ch):
            if i >= n_xch:
                continue
            # 优先保留原始合法名 (如 standard_1020 含的双极对 'FP1-F7'); 否则归一化
            raw_up = str(c).strip().upper()
            if raw_up in std_set:
                s = raw_up
            else:
                s = normalize_ch_name(c)
            if s in std_set and s not in seen:
                keep_idx.append(i); std_names.append(s); seen.add(s)
    return keep_idx, std_names


# ---- 坐标几何重建映射 (mne standard_1005最近邻, 方向一致性>0.99验证) ----
# 这些数据集通道名是占位符(EEG001/1-A1), 但原始BIDS有electrodes.tsv坐标
PERDATASET_CHMAP_DS002718 = {'EEG001': 'F9', 'EEG003': 'TP9', 'EEG009': 'F8', 'EEG010': 'F6', 'EEG011': 'FC6', 'EEG014': 'CP6', 'EEG015': 'P6', 'EEG016': 'P8', 'EEG017': 'PO10', 'EEG018': 'AF10', 'EEG019': 'AF8', 'EEG021': 'F4', 'EEG022': 'FC4', 'EEG023': 'C4', 'EEG024': 'CP4', 'EEG025': 'P4', 'EEG026': 'PO6', 'EEG027': 'O2', 'EEG030': 'FPZ', 'EEG031': 'AF2', 'EEG032': 'F2', 'EEG033': 'FCZ', 'EEG034': 'CZ', 'EEG037': 'OZ', 'EEG038': 'IZ', 'EEG041': 'FP1', 'EEG042': 'AF1', 'EEG043': 'F1', 'EEG044': 'FC1', 'EEG045': 'C1', 'EEG046': 'CP1', 'EEG047': 'P1', 'EEG048': 'PO3', 'EEG049': 'O1', 'EEG050': 'PO9', 'EEG051': 'AF9', 'EEG052': 'AF7', 'EEG053': 'F5', 'EEG054': 'F3', 'EEG055': 'FC3', 'EEG056': 'C3', 'EEG057': 'CP3', 'EEG058': 'P3', 'EEG059': 'P5', 'EEG060': 'PO7', 'EEG065': 'P9', 'EEG066': 'F7', 'EEG067': 'FC5', 'EEG068': 'C5', 'EEG069': 'CP5', 'EEG070': 'P7', 'EEG071': 'FT7', 'EEG074': 'T9'}

PERDATASET_CHMAP_DS004475 = {'1-A10': 'F7', '1-A12': 'F9', '1-A13': 'FT9', '1-A15': 'FT7', '1-A17': 'FC5', '1-A19': 'C3', '1-A20': 'CCP5', '1-A21': 'C5', '1-A22': 'T3', '1-A23': 'T7', '1-A26': 'TP9', '1-A29': 'TP7', '1-A3': 'C1', '1-A30': 'CP5', '1-A5': 'FC3', '1-A7': 'F3', '1-A8': 'F5', '1-B10': 'PO9', '1-B12': 'PO7', '1-B13': 'PO5', '1-B16': 'O1', '1-B17': 'PO3', '1-B18': 'PO1', '1-B19': 'P1', '1-B22': 'POZ', '1-B24': 'OZ', '1-B26': 'IZ', '1-B29': 'O2', '1-B3': 'P3', '1-B30': 'PO4', '1-B31': 'PO2', '1-B4': 'P5', '1-B6': 'P7', '1-C11': 'CP4', '1-C13': 'CP6', '1-C20': 'CCP6', '1-C21': 'CCP4', '1-C22': 'C4', '1-C3': 'P4', '1-C32': 'FC6', '1-C4': 'P6', '1-C5': 'PO8', '1-C7': 'PO10', '1-C9': 'P8', '1-D1': 'C2', '1-D10': 'AF4', '1-D11': 'F4', '1-D12': 'F2', '1-D13': 'FC2', '1-D14': 'FCZ', '1-D17': 'FC1', '1-D20': 'AF2', '1-D21': 'AFZ', '1-D22': 'FPZ', '1-D25': 'FP1', '1-D26': 'AF3', '1-D27': 'AF1', '1-D28': 'F1', '1-D3': 'FC4', '1-D30': 'AF5', '1-D31': 'AF7', '1-D32': 'AF9', '1-D5': 'F6', '1-D6': 'F8', '1-D8': 'AF8', '1-D9': 'AF6'}

PERDATASET_CHMAP_DS002814 = {'EEG 000': 'AFZ', 'EEG 001': 'FP1', 'EEG 002': 'FP2', 'EEG 003': 'AF3', 'EEG 004': 'AF4', 'EEG 005': 'F7', 'EEG 006': 'F3', 'EEG 007': 'FZ', 'EEG 008': 'F4', 'EEG 009': 'F8', 'EEG 010': 'FC5', 'EEG 011': 'FC1', 'EEG 012': 'FC2', 'EEG 013': 'FC6', 'EEG 014': 'T7', 'EEG 015': 'C3', 'EEG 016': 'CZ', 'EEG 017': 'C4', 'EEG 018': 'T8', 'EEG 019': 'CP5', 'EEG 020': 'CP1', 'EEG 021': 'CP2', 'EEG 022': 'CP6', 'EEG 023': 'P7', 'EEG 024': 'P3', 'EEG 025': 'PZ', 'EEG 026': 'P4', 'EEG 027': 'P8', 'EEG 028': 'PO3', 'EEG 029': 'PO4', 'EEG 030': 'O1', 'EEG 031': 'O2', 'EEG 032': 'AF7', 'EEG 033': 'AF8', 'EEG 034': 'F5', 'EEG 035': 'F1', 'EEG 036': 'F2', 'EEG 037': 'F6', 'EEG 038': 'FT7', 'EEG 039': 'FC3', 'EEG 040': 'FCZ', 'EEG 041': 'FC4', 'EEG 042': 'FT8', 'EEG 043': 'C5', 'EEG 044': 'C1', 'EEG 045': 'C2', 'EEG 046': 'C6', 'EEG 047': 'TP7', 'EEG 048': 'CP3', 'EEG 049': 'CPZ', 'EEG 050': 'CP4', 'EEG 051': 'TP8', 'EEG 052': 'P5', 'EEG 053': 'P1', 'EEG 054': 'P2', 'EEG 055': 'P6', 'EEG 056': 'PO7', 'EEG 057': 'POZ', 'EEG 058': 'PO8', 'EEG 059': 'OZ', 'EEG 060': 'P9', 'EEG 061': 'P10', 'EEG 062': 'TP9', 'EEG000': 'AFZ', 'EEG001': 'FP1', 'EEG002': 'FP2', 'EEG003': 'AF3', 'EEG004': 'AF4', 'EEG005': 'F7', 'EEG006': 'F3', 'EEG007': 'FZ', 'EEG008': 'F4', 'EEG009': 'F8', 'EEG010': 'FC5', 'EEG011': 'FC1', 'EEG012': 'FC2', 'EEG013': 'FC6', 'EEG014': 'T7', 'EEG015': 'C3', 'EEG016': 'CZ', 'EEG017': 'C4', 'EEG018': 'T8', 'EEG019': 'CP5', 'EEG020': 'CP1', 'EEG021': 'CP2', 'EEG022': 'CP6', 'EEG023': 'P7', 'EEG024': 'P3', 'EEG025': 'PZ', 'EEG026': 'P4', 'EEG027': 'P8', 'EEG028': 'PO3', 'EEG029': 'PO4', 'EEG030': 'O1', 'EEG031': 'O2', 'EEG032': 'AF7', 'EEG033': 'AF8', 'EEG034': 'F5', 'EEG035': 'F1', 'EEG036': 'F2', 'EEG037': 'F6', 'EEG038': 'FT7', 'EEG039': 'FC3', 'EEG040': 'FCZ', 'EEG041': 'FC4', 'EEG042': 'FT8', 'EEG043': 'C5', 'EEG044': 'C1', 'EEG045': 'C2', 'EEG046': 'C6', 'EEG047': 'TP7', 'EEG048': 'CP3', 'EEG049': 'CPZ', 'EEG050': 'CP4', 'EEG051': 'TP8', 'EEG052': 'P5', 'EEG053': 'P1', 'EEG054': 'P2', 'EEG055': 'P6', 'EEG056': 'PO7', 'EEG057': 'POZ', 'EEG058': 'PO8', 'EEG059': 'OZ', 'EEG060': 'P9', 'EEG061': 'P10', 'EEG062': 'TP9'}

PERDATASET_CHMAP_DS004460 = {'G11': 'F8', 'G12': 'FT8', 'G14': 'T10', 'G20': 'TP8', 'G21': 'T6', 'G22': 'P8', 'G23': 'PO10', 'G24': 'PO6', 'G25': 'PO8', 'G28': 'PO4', 'G29': 'O2', 'G4': 'AF4', 'G7': 'AF8', 'G8': 'F6', 'G9': 'F10', 'N10': 'F9', 'N11': 'FT9', 'N12': 'T9', 'N14': 'P9', 'N20': 'TP9', 'N32': 'C2', 'N4': 'FT7', 'N5': 'T3', 'R12': 'FCZ', 'R14': 'FC2', 'R16': 'CZ', 'R18': 'CP2', 'R19': 'CCP1', 'R2': 'AF2', 'R20': 'CPZ', 'R21': 'P2', 'R23': 'C1', 'R24': 'CP1', 'R26': 'PZ', 'R29': 'P1', 'R30': 'PO1', 'R31': 'POZ', 'R32': 'OZ', 'R4': 'F1', 'R6': 'F2', 'R7': 'FZ', 'R8': 'FC1', 'W11': 'F3', 'W12': 'FC3', 'W16': 'CCP3', 'W17': 'C3', 'W18': 'CP3', 'W20': 'P3', 'W21': 'PO3', 'W22': 'C5', 'W23': 'CCP5', 'W24': 'CP5', 'W25': 'P5', 'W26': 'PO5', 'W28': 'T7', 'W29': 'TP7', 'W30': 'P7', 'W6': 'F5', 'W8': 'FC5', 'Y10': 'FC6', 'Y20': 'CCP4', 'Y23': 'P6', 'Y25': 'CP4', 'Y29': 'CCP2', 'Y31': 'P4', 'Y32': 'PO2', 'Y5': 'F4', 'Y7': 'FC4'}

# ---- BCI Competition IV-2a 专属映射 (EEG-数字编号→标准10-20, 官方22通道蒙太奇顺序) ----
# 文件命名 EEG-Fz/EEG-0../EEG-16 部分数字化, 但顺序=官方22电极标准蒙太奇.
# 锚点 EEG-Fz/C3/Cz/C4/Pz 位置全对齐官方顺序验证通过. EOG-* 自动排除(不在std_1020).
PERDATASET_CHMAP_BCICV_2A = {
    'EEG-FZ': 'FZ', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCZ', 'EEG-3': 'FC2',
    'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-CZ': 'CZ',
    'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1',
    'EEG-11': 'CPZ', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-PZ': 'PZ',
    'EEG-15': 'P2', 'EEG-16': 'POZ',
}


PERDATASET_CHMAP_DS004395 = {'A11': 'PO9', 'A16': 'O1', 'A18': 'PO3', 'A19': 'POZ', 'A2': 'CPZ', 'A21': 'OZ', 'A24': 'IZ', 'A29': 'O2', 'A3': 'PZ', 'A31': 'PO4', 'A32': 'PO2', 'A5': 'PO1', 'A6': 'P3', 'A8': 'PO5', 'A9': 'PO7', 'B11': 'T6', 'B12': 'P8', 'B13': 'P6', 'B14': 'TP8', 'B15': 'TPP8H', 'B16': 'CP6', 'B18': 'CP4', 'B2': 'CP2', 'B20': 'CCP2', 'B22': 'CCP4', 'B24': 'CCP6', 'B27': 'T8', 'B29': 'C6', 'B3': 'P4', 'B31': 'C4', 'B5': 'PO6', 'B6': 'PO8', 'B8': 'PO10', 'C10': 'F4', 'C11': 'FC2', 'C13': 'F2', 'C15': 'AF4', 'C16': 'AF6', 'C17': 'FPZ', 'C18': 'AF2', 'C2': 'C2', 'C20': 'FZ', 'C22': 'FCZ', 'C23': 'CZ', 'C24': 'FC1', 'C26': 'F1', 'C28': 'AF3', 'C29': 'AF5', 'C31': 'F5', 'C32': 'F3', 'C4': 'FC4', 'C5': 'FC6', 'C7': 'FT8', 'C9': 'F6', 'D10': 'C5', 'D12': 'C3', 'D14': 'CCP1', 'D16': 'CP1', 'D19': 'CCP3', 'D2': 'C1', 'D21': 'CCP5', 'D22': 'TTP7H', 'D24': 'TP7', 'D26': 'CP5', 'D28': 'CP3', 'D29': 'P5', 'D30': 'P7', 'D31': 'T5', 'D4': 'FC3', 'D5': 'FC5', 'D8': 'T7'}


# EESM23 labels encode side and top/bottom around each ear; LC is the common reference.
PERDATASET_CHMAP_DS005178 = {
    'RB': 'A2', 'RT': 'T10', 'LB': 'A1', 'LT': 'T9',
}


PERDATASET_CHMAP_DS005555 = {
    'HB_1': 'AF7',
    'HB_2': 'AF8',
}


PERDATASET_CHMAPS = {
    'ds004395': PERDATASET_CHMAP_DS004395,
    'ds002718': PERDATASET_CHMAP_DS002718,
    'ds004475': PERDATASET_CHMAP_DS004475,
    'ds002814': PERDATASET_CHMAP_DS002814,
    'ds004460': PERDATASET_CHMAP_DS004460,
    'ds005178': PERDATASET_CHMAP_DS005178,
    'ds005555': PERDATASET_CHMAP_DS005555,
    'BCICV_2a': PERDATASET_CHMAP_BCICV_2A,
}


def get_perdataset_chmap(dataset_id):
    """按数据集名取坐标重建映射 (占位通道名→标准10-20). 无则None."""
    base = str(dataset_id).replace('ON_', '').strip()
    for k, v in PERDATASET_CHMAPS.items():
        if base == k or base.startswith(k):
            return v
    return None
