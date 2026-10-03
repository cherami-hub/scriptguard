# -*- coding: utf-8 -*-
"""
rules.py
Script-aware security rule library (minimal viable version of the
five-level fraud-script ontology in Sec. 4.2 of the PhD proposal).

Each script: list of (pattern, weight). Substring/regex match on raw text.
Cross-cutting inducement-action patterns and structural features included.

NOTE: IL_Political_propaganda intentionally has NO active keywords
(sensitive category handled only by the statistical model).
"""
import re

# ---------------- script keyword rules ----------------
# weight ~ discriminative strength (hand-set, tunable on train)
SCRIPT_RULES = {
    "FR_Phishing_Bank": [
        ("密码器", 3), ("口令卡", 3), ("U盾", 3), ("电子银行", 2),
        ("积分兑换现金", 3), ("积分", 1), ("兑换", 1), ("逾期失效", 2),
        ("账户冻结", 3), ("账户异常", 3), ("解冻", 2), ("实名认证", 2),
        ("登陆我行", 3), ("登录我行", 3), ("手机银行", 1), ("网上银行", 1),
        ("建设银行", 2), ("建行", 2), ("工商银行", 2), ("农业银行", 2),
        ("中国银行", 2), ("招商银行", 2), ("兴业银行", 2), ("浦发银行", 2),
        ("交通银行", 2), ("邮政储蓄", 2), ("光大银行", 2), ("民生银行", 2),
        ("中信银行", 2), ("平安银行", 2), ("信用卡", 1), ("储蓄卡", 1),
        ("核实身份", 2), ("安全账户", 3), ("系统升级", 1), ("证书", 2),
    ],
    "FR_Phishing_Other": [
        ("话费", 2), ("积分兑换话费", 3), ("充值", 1), ("套餐", 1),
        ("会员到期", 2), ("续费", 1), ("退订", 1), ("流量包", 1),
        ("营业厅", 2), ("联通", 2), ("移动公司", 2), ("电信营业厅", 2),
        ("宽带", 1), ("话费余额", 2), ("话费充值", 2),
    ],
    "FR_Financial": [
        ("股票", 2), ("荐股", 3), ("涨停", 3), ("内幕消息", 3), ("私募", 3),
        ("配资", 3), ("牛股", 3), ("建仓", 2), ("带单", 3), ("老师指导", 2),
        ("炒黄金", 3), ("炒白银", 3), ("炒原油", 3), ("炒外汇", 3),
        ("现货", 2), ("期货", 2), ("大盘", 1), ("个股", 2), ("拉升", 2),
        ("收益", 1), ("稳赚", 2), ("翻倍", 2),
    ],
    "FR_Other": [
        ("中奖", 3), ("幸运观众", 3), ("抽奖", 2), ("领取奖品", 3),
        ("栏目组", 3), ("节目组", 3), ("法院传票", 3), ("涉嫌犯罪", 3),
        ("包裹未领", 3), ("邮局通知", 2), ("社保卡异常", 3), ("医保停用", 3),
        ("补贴", 1), ("退款", 2), ("机票改签", 3), ("航班取消", 3),
        ("助学金", 2), ("退税", 2), ("包裹", 1),
    ],
    "IL_Gambling": [
        ("百家乐", 3), ("六合彩", 3), ("六合", 2), ("时时彩", 3), ("彩票", 1),
        ("投注", 2), ("博彩", 3), ("娱乐城", 3), ("真人荷官", 3), ("荷官", 3),
        ("棋牌", 2), ("捕鱼", 2), ("赌球", 3), ("赌", 2), ("开奖", 2),
        ("赔率", 2), ("首存", 3), ("注册送", 2), ("皇冠", 2), ("金沙", 2),
        ("威尼斯人", 3), ("太阳城", 3), ("下注", 3), ("盘口", 2),
    ],
    "IL_Escort_service": [
        ("小姐", 3), ("上门服务", 3), ("包夜", 3), ("休闲会所", 2),
        ("按摩", 2), ("模特兼职", 2), ("大学生兼职", 2), ("全套服务", 3),
        ("桑拿", 2), ("会所", 1),
    ],
    "IL_Fake_ID_and_invoice": [
        ("发票", 3), ("代开发票", 3), ("增值税发票", 3), ("刻章", 3),
        ("办证件", 3), ("文凭", 3), ("学历", 2), ("驾驶证", 2),
        ("营业执照", 2), ("公章", 2), ("办证", 2), ("真票", 2),
    ],
    "IL_Political_propaganda": [],   # intentionally empty (see docstring)
}

# ---------------- cross-cutting inducement actions ----------------
INDUCEMENT_PATTERNS = [
    r"URL", r"www\.", r"\.com", r"\.cn", r"\.net", r"http",
    r"点击", r"登陆", r"登录", r"下载", r"安装",
    r"客服", r"咨询", r"联系", r"拨打", r"致电", r"来电",
    r"微信", r"扣扣", r"QQ", r"威信",
    r"转账", r"汇款", r"密码", r"验证码", r"卡号", r"账号",
]

_COMPILED_IND = [re.compile(p) for p in INDUCEMENT_PATTERNS]
_COMPILED_RULES = {
    k: [(re.compile(re.escape(p)), w) for p, w in v]
    for k, v in SCRIPT_RULES.items()
}

TOP_OF = {k: k.split("_")[0] for k in SCRIPT_RULES}
FEATURE_NAMES = (
    [f"script:{k}" for k in SCRIPT_RULES]
    + ["inducement_hits", "has_url", "has_contact", "has_money_term",
       "rule_margin", "msg_len"]
)

# ---------------- feature extraction ----------------
def script_scores(text):
    """Weighted match score per script category."""
    scores = {}
    for k, rules in _COMPILED_RULES.items():
        s = 0
        for pat, w in rules:
            if pat.search(text):
                s += w
        scores[k] = s
    return scores

def inducement_count(text):
    return sum(1 for p in _COMPILED_IND if p.search(text))

def rule_features(text):
    """Dense feature vector (len = len(FEATURE_NAMES))."""
    sc = script_scores(text)
    fr = sum(v for k, v in sc.items() if TOP_OF[k] == "FR")
    il = sum(v for k, v in sc.items() if TOP_OF[k] == "IL")
    ind = inducement_count(text)
    has_url = 1 if re.search(r"URL|www\.|\.com|\.cn|http", text) else 0
    has_contact = 1 if re.search(r"客服|联系|拨打|致电|微信|QQ|扣扣", text) else 0
    has_money = 1 if re.search(r"转账|汇款|密码|验证码|卡号|现金|元", text) else 0
    vec = [sc[k] for k in SCRIPT_RULES]
    vec += [ind, has_url, has_contact, has_money, (fr + il) - 0, len(text)]
    return vec

def rule_predict_top(text, ad_prior=1.0):
    """Rule-only baseline: fraud(FR/IL) if weighted fraud evidence > 0,
    else AD. Returns ('FR'/'IL'/'AD', margin)."""
    sc = script_scores(text)
    fr = sum(v for k, v in sc.items() if TOP_OF[k] == "FR")
    il = sum(v for k, v in sc.items() if TOP_OF[k] == "IL")
    if fr + il > 0:
        bonus = 0.5 * inducement_count(text)
        return ("FR" if fr >= il else "IL"), fr + il + bonus
    return "AD", 0.0

if __name__ == "__main__":
    demo = "尊敬的客户您的建行卡积分已满万分可兑换现金礼包访问手机网URL查询兑换逾期失效"
    print(script_scores(demo))
    print(rule_predict_top(demo))
    print(dict(zip(FEATURE_NAMES, rule_features(demo))))
