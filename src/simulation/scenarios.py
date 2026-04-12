"""Preset scenarios with sample users and contents based on KuaiRec distributions.

These provide a quick-start environment before loading the full KuaiRec dataset.
User profiles and content items are modeled after real KuaiRec feature distributions.
"""

from __future__ import annotations

from src.models import ContentItem, Gender, UserProfile


def create_sample_users(n: int = 20) -> list[UserProfile]:
    """Create sample users based on KuaiRec demographic distributions."""
    templates = [
        UserProfile("u001", Gender.FEMALE, "18-24", "广东", "广州", "一线", ["美妆", "穿搭"], "high", "high"),
        UserProfile("u002", Gender.MALE, "25-30", "北京", "北京", "一线", ["科技", "游戏"], "high", "high"),
        UserProfile("u003", Gender.FEMALE, "25-30", "浙江", "杭州", "二线", ["美食", "旅行"], "mid", "medium"),
        UserProfile("u004", Gender.MALE, "18-24", "四川", "成都", "二线", ["搞笑", "体育"], "mid", "high"),
        UserProfile("u005", Gender.FEMALE, "31-40", "江苏", "南京", "二线", ["育儿", "美食"], "mid", "medium"),
        UserProfile("u006", Gender.MALE, "31-40", "上海", "上海", "一线", ["财经", "汽车"], "high", "low"),
        UserProfile("u007", Gender.FEMALE, "18-24", "湖北", "武汉", "二线", ["宠物", "音乐"], "low", "high"),
        UserProfile("u008", Gender.MALE, "18-24", "河南", "郑州", "三线", ["游戏", "动漫"], "low", "high"),
        UserProfile("u009", Gender.FEMALE, "25-30", "山东", "青岛", "三线", ["健身", "美食"], "mid", "medium"),
        UserProfile("u010", Gender.MALE, "41-50", "广东", "深圳", "一线", ["钓鱼", "新闻"], "high", "low"),
        UserProfile("u011", Gender.FEMALE, "25-30", "福建", "厦门", "三线", ["宠物", "穿搭"], "mid", "medium"),
        UserProfile("u012", Gender.MALE, "25-30", "浙江", "温州", "三线", ["创业", "科技"], "mid", "medium"),
        UserProfile("u013", Gender.FEMALE, "18-24", "湖南", "长沙", "二线", ["追剧", "美妆"], "low", "high"),
        UserProfile("u014", Gender.MALE, "31-40", "江苏", "苏州", "二线", ["摄影", "旅行"], "high", "medium"),
        UserProfile("u015", Gender.FEMALE, "41-50", "河北", "石家庄", "三线", ["广场舞", "养生"], "low", "medium"),
        UserProfile("u016", Gender.MALE, "18-24", "陕西", "西安", "二线", ["说唱", "篮球"], "mid", "high"),
        UserProfile("u017", Gender.FEMALE, "25-30", "重庆", "重庆", "二线", ["美食", "探店"], "mid", "high"),
        UserProfile("u018", Gender.MALE, "25-30", "广东", "东莞", "三线", ["健身", "汽车"], "mid", "medium"),
        UserProfile("u019", Gender.FEMALE, "31-40", "北京", "北京", "一线", ["亲子", "教育"], "high", "medium"),
        UserProfile("u020", Gender.MALE, "41-50", "山东", "济南", "三线", ["三农", "钓鱼"], "low", "low"),
    ]
    return templates[:n]


def create_sample_contents() -> list[ContentItem]:
    """Create sample content items based on KuaiRec category taxonomy."""
    return [
        # 宠物类
        ContentItem("v001", "我家柯基又在拆家了 #宠物日常 #柯基", ["宠物日常", "柯基"], "宠物", "宠物日常记录", "宠物狗", 25),
        ContentItem("v002", "猫咪第一次见到雪的反应太搞笑了 #猫咪", ["猫咪", "搞笑"], "宠物", "宠物日常记录", "宠物猫", 18),
        ContentItem("v003", "教你如何给狗狗洗澡不被咬 #宠物护理", ["宠物护理", "教程"], "宠物", "宠物知识", "宠物护理", 45),
        # 美食类
        ContentItem("v004", "3分钟学会正宗四川麻婆豆腐 #美食教程 #川菜", ["美食教程", "川菜"], "美食", "美食教程", "家常菜", 180),
        ContentItem("v005", "探店！成都最火的串串香到底有多好吃 #探店", ["探店", "成都美食"], "美食", "探店", "火锅串串", 120),
        ContentItem("v006", "今天做了颜值超高的草莓蛋糕 #烘焙 #甜点", ["烘焙", "甜点"], "美食", "美食教程", "烘焙甜点", 90),
        # 搞笑类
        ContentItem("v007", "当代大学生的真实写照哈哈哈 #搞笑 #大学生", ["搞笑", "大学生"], "搞笑", "幽默段子", "校园搞笑", 15),
        ContentItem("v008", "方言版配音太绝了 #搞笑配音 #方言", ["搞笑配音", "方言"], "搞笑", "搞笑配音", "方言搞笑", 30),
        # 科技类
        ContentItem("v009", "2026年最值得买的5款手机推荐 #科技 #手机", ["科技", "手机推荐"], "科技", "数码测评", "手机", 240),
        ContentItem("v010", "AI编程效率提升10倍的技巧分享 #AI #编程", ["AI", "编程"], "科技", "技术分享", "AI应用", 180),
        # 美妆穿搭类
        ContentItem("v011", "夏日清透妆容教程 5分钟出门 #美妆 #夏日妆容", ["美妆", "夏日"], "美妆", "化妆教程", "日常妆容", 150),
        ContentItem("v012", "小个子女生显高穿搭秘籍 #穿搭 #显高", ["穿搭", "小个子"], "穿搭", "穿搭技巧", "女生穿搭", 60),
        # 健身类
        ContentItem("v013", "居家15分钟腹肌训练 无需器材 #健身 #腹肌", ["健身", "腹肌"], "健身", "健身教程", "居家健身", 900),
        ContentItem("v014", "增肌期一日三餐怎么吃 #健身餐 #增肌", ["健身餐", "增肌"], "健身", "健身饮食", "增肌饮食", 120),
        # 育儿/亲子
        ContentItem("v015", "宝宝辅食这样做营养又好吃 #育儿 #辅食", ["育儿", "辅食"], "育儿", "育儿知识", "婴儿辅食", 90),
        # 三农/生活
        ContentItem("v016", "农村大爷教你种出又大又甜的西瓜 #三农 #种植", ["三农", "种植"], "三农", "农业技术", "瓜果种植", 200),
    ]


def create_pet_scenario() -> tuple[list[UserProfile], list[ContentItem]]:
    """Create a focused scenario for pet content recommendation."""
    users = create_sample_users(20)
    contents = [c for c in create_sample_contents() if c.category_l1 == "宠物"]
    return users, contents


def create_food_scenario() -> tuple[list[UserProfile], list[ContentItem]]:
    """Create a focused scenario for food content recommendation."""
    users = create_sample_users(20)
    contents = [c for c in create_sample_contents() if c.category_l1 == "美食"]
    return users, contents


def create_full_scenario() -> tuple[list[UserProfile], list[ContentItem]]:
    """Create the full scenario with all categories."""
    return create_sample_users(20), create_sample_contents()
