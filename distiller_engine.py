import json
import re
import os  # 新增：用于文件和目录操作
from collections import Counter


class PersonaDistiller:
    """人格蒸馏引擎：从原始聊天记录中提取语言风格和情感特征"""

    def __init__(self):
        # 用于匹配常见的 Emoji 表情
        self.emoji_pattern = re.compile(r"[\u263a-\U0001f64f]")

    def distill(self, raw_text: str) -> dict:
        """蒸馏逻辑：提取口癖、回答倾向、表情包偏好"""
        # 过滤掉空行
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # 1. 提取口癖 (简单的频率分析)
        words = re.findall(r'\w+', raw_text)
        # 排除单个字的语气词干扰，提取排名前 10 的词汇
        common_words = [w for w, c in Counter(words).most_common(10) if len(w) > 1]

        # 2. 回答长度倾向
        lengths = [len(line) for line in lines]
        avg_len = sum(lengths) / len(lengths) if lengths else 0

        # 3. 表情包分析
        emojis = self.emoji_pattern.findall(raw_text)
        top_emojis = [e for e, c in Counter(emojis).most_common(5)]

        return {
            "avg_reply_length": round(avg_len, 1),
            "top_keywords": common_words,
            "favorite_emojis": top_emojis,
            "style_tags": self._infer_tags(avg_len, top_emojis),
            "corpus_sample": lines[:50]  # 保留前 50 条作为大模型的 Few-shot 样本
        }

    def _infer_tags(self, avg_len: float, emojis: list) -> list:
        """补充漏掉的内部方法：根据数据推断性格标签"""
        tags = []
        if avg_len < 10:
            tags.append("高冷/精简")
        elif avg_len > 40:
            tags.append("长篇大论/倾诉欲强")

        if len(emojis) >= 3:
            tags.append("表情包大户/情绪外放")
        elif len(emojis) == 0:
            tags.append("严肃/不爱用表情")

        if not tags:
            tags.append("语气平和")
        return tags

    def save_to_skills(self, raw_text: str, persona_name: str, skills_dir: str = "skills"):
        """将蒸馏结果保存到技能目录中"""
        # 1. 执行蒸馏
        distilled_data = self.distill(raw_text)

        # 2. 创建专属文件夹 (例如: skills/ex_girlfriend)
        target_dir = os.path.join(skills_dir, persona_name)
        os.makedirs(target_dir, exist_ok=True)

        # 3. 保存语料库文件 (corpus.json)
        corpus_path = os.path.join(target_dir, "corpus.json")
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(distilled_data, f, ensure_ascii=False, indent=2)

        # 4. 生成系统所需的 SKILL.md 配置文件
        # 注意这里的 YAML 缩进必须顶格
        skill_md_content = f"""---
name: {persona_name}
description: 人格注入配置：{persona_name}
type: persona
corpus_file: corpus.json
---
"""
        md_path = os.path.join(target_dir, "SKILL.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(skill_md_content)

        print(f"✅ 人格 [{persona_name}] 蒸馏完成！")
        print(f"📁 已生成配置文件：{md_path}")
        print(f"📁 已生成灵魂数据：{corpus_path}")


# ==========================================
# 实际运行脚本
# ==========================================
if __name__ == "__main__":
    file_path = "raw_chat.txt"

    if not os.path.exists(file_path):
        print(f"❌ 找不到文件：{file_path}，请确认文件放在了正确的位置！")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            chat_content = f.read()

        # 实例化蒸馏器
        distiller = PersonaDistiller()

        # 执行蒸馏
        print("开始进行灵魂蒸馏...")

        # 【关键修改】：这里直接写 "skills"，它就会在根目录下生成了！
        distiller.save_to_skills(
            raw_text=chat_content,
            persona_name="her",
            skills_dir="skills"
        )