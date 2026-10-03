# -*- coding: utf-8 -*-
"""
variants.py
Adversarial text-variant generator for Chinese fraud-diversion messages,
plus a normalization layer that reverses structural variants.

Variant taxonomy (8 types), grounded in observed FBS smuggling tactics:
  T1 homophone     谐音字替换 (same-pinyin substitution, corpus-derived)
  T2 pinyin_init   拼音首字母化 (微信 -> wx)
  T3 symbol_insert 符号插入    (微信 -> 微-信 / 微*信 / 微 信)
  T4 traditional   繁体替换    (银行 -> 銀行, high-freq fraud vocab map)
  T5 lookalike     形近字/leet (日->曰, 大->太, 0->O)
  T6 fullwidth     全角化      (vip -> ｖｉｐ)
  T7 emoji_noise   表情/装饰符号噪声插入
  T8 deletion      字符脱落    (微信 -> 微)
"""
import random
from functools import lru_cache
from pypinyin import lazy_pinyin, Style

@lru_cache(maxsize=20000)
def _py(ch):
    return lazy_pinyin(ch, style=Style.NORMAL)[0]

@lru_cache(maxsize=20000)
def _py_first(ch):
    return lazy_pinyin(ch, style=Style.FIRST_LETTER)[0]

# ---------- maps ----------
TRAD = dict(zip(
    "银行钱发积兑换账户证证书过期失效登陆录网网站客服电话费充值会员积分礼包奖金奖品领取冻结解冻风险安全升级认证核实请点击下载链接赌博彩票投注娱乐发票代开刻章证件学历兼职服务内容小姐",
    "銀行錢發積兌換賬戶證證書過期失效登陸錄網網站客服電話費充值會員積分禮包獎金獎品領取凍結解凍風險安全升級認證核實請點擊下載鏈接賭博彩票投註娛樂發票代開刻章證件學歷兼職服務內容小姐"
))
TRAD_INV = {v: k for k, v in TRAD.items()}

LOOKALIKE = {
    "日": "曰", "大": "太", "人": "入", "未": "末", "土": "士",
    "王": "玉", "己": "已", "拨": "拔", "免": "兔", "辨": "辩",
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
    "o": "0", "l": "1", "O": "0",
}

INSERT_SYMBOLS = ["-", "*", ".", " ", "~", "/", "#", "@", "!", "^"]
EMOJI_SYMBOLS = ["★", "※", "【", "】", "→", "■", "●", "▲", "◆", "♦", "☎", "♨", "✿", "❀", "～", "§"]

def _is_cjk(ch):
    return "一" <= ch <= "鿿"

def _fw(ch):
    o = ord(ch)
    if 33 <= o <= 126:
        return chr(o + 0xFEE0)
    return ch

def _hw(ch):
    o = ord(ch)
    if o == 0x3000:
        return " "
    if 0xFF01 <= o <= 0xFF5E:
        return chr(o - 0xFEE0)
    return ch

# ---------- homophone index (built lazily from corpus vocab) ----------
_PINYIN_INDEX = None

# curated homophone pairs observed in real fraud-SMS evasion
CURATED_HOMO = {
    "微": ["威", "薇", "味"], "信": ["芯", "欣", "辛"], "银": ["垠", "吟"],
    "账": ["帐", "丈"], "积": ["机", "基", "鸡"], "分": ["份", "芬"],
    "兑": ["对", "队"], "换": ["幻", "唤"], "客": ["课", "克"],
    "服": ["福", "扶", "浮"], "赌": ["堵", "肚"], "博": ["搏", "薄"],
    "票": ["飘", "漂"], "彩": ["采", "踩"], "钱": ["前", "潜"],
    "发": ["法"], "证": ["正", "政"], "章": ["张", "彰"],
    "贷": ["带", "代"], "款": ["宽"], "利": ["力", "立", "丽"],
    "率": ["绿", "滤"], "赚": ["转", "专"], "金": ["今", "斤", "巾"],
    "卡": ["咔", "佧"], "刷": ["唰"], "单": ["丹", "担"],
    "返": ["反", "贩"], "现": ["线", "限", "县"], "提": ["题", "啼"],
    "密": ["蜜", "秘"], "码": ["马", "玛"], "验": ["燕", "艳"],
    "冻": ["动", "洞"], "结": ["节", "杰"], "案": ["按", "暗"],
    "局": ["拘", "菊"], "检": ["简", "减"], "法": ["发", "罚"],
    "关": ["官", "观"], "注": ["住", "助"], "册": ["侧", "测"],
    "送": ["松", "宋"], "礼": ["里", "李", "理"], "包": ["胞", "炮"],
    "奖": ["讲", "蒋"], "中": ["忠", "终"], "领": ["岭", "令"],
}

def build_homophone_index(vocab_chars):
    """vocab_chars: iterable of Chinese chars appearing in the corpus."""
    global _PINYIN_INDEX
    idx = {}
    for ch in set(vocab_chars):
        if not _is_cjk(ch):
            continue
        py = _py(ch)
        idx.setdefault(py, []).append(ch)
    _PINYIN_INDEX = {k: v for k, v in idx.items() if len(v) > 1}
    # merge curated pairs (same-pinyin by construction)
    for ch, subs in CURATED_HOMO.items():
        if not subs:
            continue
        py = _py(ch)
        merged = set(_PINYIN_INDEX.get(py, [])) | set(subs) | {ch}
        _PINYIN_INDEX[py] = sorted(merged)
    return len(_PINYIN_INDEX)

def _homophone_sub(ch, rng):
    if _PINYIN_INDEX is None or not _is_cjk(ch):
        return None
    py = _py(ch)
    cands = _PINYIN_INDEX.get(py)
    if not cands:
        return None
    for _ in range(4):
        c = rng.choice(cands)
        if c != ch:
            return c
    return None

# ---------- per-char perturbation ----------
PERTURBATION_TYPES = [
    "homophone", "pinyin_init", "symbol_insert", "traditional",
    "lookalike", "fullwidth", "emoji_noise", "deletion",
]

def perturb(text, intensity, rng, types=None):
    """Perturb each eligible char with probability `intensity`.
    types: subset of PERTURBATION_TYPES to use (None = all)."""
    types = types or PERTURBATION_TYPES
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if rng.random() < intensity and (_is_cjk(ch) or ch.isalnum()):
            t = rng.choice(types)
            if t == "homophone":
                sub = _homophone_sub(ch, rng)
                out.append(sub if sub else ch)
            elif t == "pinyin_init" and _is_cjk(ch):
                py = _py_first(ch)
                out.append(py)
            elif t == "symbol_insert":
                out.append(ch)
                out.append(rng.choice(INSERT_SYMBOLS))
            elif t == "traditional":
                out.append(TRAD.get(ch, ch))
            elif t == "lookalike":
                out.append(LOOKALIKE.get(ch, ch))
            elif t == "fullwidth":
                out.append(_fw(ch))
            elif t == "emoji_noise":
                out.append(ch)
                out.append(rng.choice(EMOJI_SYMBOLS))
            elif t == "deletion":
                pass  # drop char
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)

# ---------- keyword-targeted attack (realistic evasion model) ----------
_KEYWORD_RE = None

def set_keywords(keywords):
    """Provide the evasion-target keyword set (fraud-script vocabulary).
    Builds a combined regex for span matching."""
    global _KEYWORD_RE
    import re
    kws = sorted({k for k in keywords if k}, key=len, reverse=True)
    _KEYWORD_RE = re.compile("|".join(re.escape(k) for k in kws)) if kws else None
    return len(kws)

def perturb_keywords(text, p, rng, types=None, bg_ratio=0.1):
    """Keyword-targeted perturbation: chars inside fraud-keyword spans are
    perturbed with probability p; background chars with p*bg_ratio.
    This mirrors real evasion: attackers obfuscate the trigger words."""
    if _KEYWORD_RE is None:
        return perturb(text, p, rng, types)
    in_span = [False] * len(text)
    for m in _KEYWORD_RE.finditer(text):
        for i in range(m.start(), m.end()):
            in_span[i] = True
    out = []
    for i, ch in enumerate(text):
        prob = p if in_span[i] else p * bg_ratio
        if rng.random() < prob and (_is_cjk(ch) or ch.isalnum()):
            t = rng.choice(types or PERTURBATION_TYPES)
            out.append(_apply(t, ch, rng))
        else:
            out.append(ch)
    return "".join(out)

def _apply(t, ch, rng):
    if t == "homophone":
        sub = _homophone_sub(ch, rng)
        return sub if sub else ch
    if t == "pinyin_init" and _is_cjk(ch):
        return _py_first(ch)
    if t == "symbol_insert":
        return ch + rng.choice(INSERT_SYMBOLS)
    if t == "traditional":
        return TRAD.get(ch, ch)
    if t == "lookalike":
        return LOOKALIKE.get(ch, ch)
    if t == "fullwidth":
        return _fw(ch)
    if t == "emoji_noise":
        return ch + rng.choice(EMOJI_SYMBOLS)
    if t == "deletion":
        return ""
    return ch

# ---------- normalization layer (defense) ----------
_STRIP_SET = set(INSERT_SYMBOLS + EMOJI_SYMBOLS + ["　"])

def normalize(text):
    """Reverse structural variants: fullwidth->ascii, strip inserted
    symbols/emoji, traditional->simplified (fraud-vocab map)."""
    buf = []
    for ch in text:
        ch = _hw(ch)
        if ch in _STRIP_SET:
            continue
        ch = TRAD_INV.get(ch, ch)
        buf.append(ch)
    return "".join(buf)

if __name__ == "__main__":
    rng = random.Random(42)
    demo = "尊敬的客户您的银行卡积分已满万分可兑换现金礼包请登陆网站查询兑换逾期失效"
    chars = set(demo) | set("微信银钱发账户证客服赌博彩票投注娱乐发票")
    build_homophone_index(chars)
    for p in (0.1, 0.3, 0.5):
        v = perturb(demo, p, rng)
        print(f"p={p}: {v}")
        print(f"   norm: {normalize(v)}")
