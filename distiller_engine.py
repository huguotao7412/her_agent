import json
import re
import os
from collections import Counter
from pathlib import Path

class PersonaDistiller:
    """人格蒸馏引擎：从原始聊天记录中提取语言风格和情感特征"""

    def __init__(self):
        # 匹配常见 Emoji 表情
        self.emoji_pattern = re.compile(r"[\u263a-\U0001f64f]")
        # 预设的检测口癖词库
        self.tone_words = ["哈哈", "哎呀", "无语", "卧槽", "嗯嗯", "哦哦", "切", "咋了", "宝", "尊嘟", "捏", "无语子", "绝了"]

    def distill(self, raw_text: str) -> tuple[dict, dict]:
        """双轨蒸馏：提取事实语料库 + 语气情绪画像"""
        # 过滤空行
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        # ========================================
        # 轨道一：事实语料 (Fact & Corpus)
        # ========================================
        corpus_data = {
            "sample_chats": lines[:50] if len(lines) > 50 else lines
        }

        # ========================================
        # 轨道二：情绪与口癖画像 (Tone Profile)
        # ========================================
        # 1. 提取常见语气词
        found_tones = []
        for tw in self.tone_words:
            if tw in raw_text:
                found_tones.append(tw)

        # 2. 提取表情包偏好
        emojis = self.emoji_pattern.findall(raw_text)
        top_emojis = [e for e, c in Counter(emojis).most_common(5)]

        # 3. 提取标点习惯
        punct_style = []
        if "~" in raw_text or "～" in raw_text:
            punct_style.append("非常爱用波浪号(~)结尾，显得慵懒、随意或撒娇")
        if "..." in raw_text or "。。" in raw_text:
            punct_style.append("喜欢用省略号(...)或句号(。。)表达无语、延宕语气")
        if not re.search(r'[。！？]', "\n".join(lines[:50])):
            punct_style.append("极少使用正规标点符号，习惯用空格或直接发送断句")

        # 4. 回答长度倾向
        avg_len = sum(len(line) for line in lines) / len(lines) if lines else 0

        tone_profile = {
            "common_particles": found_tones,
            "favorite_emojis": top_emojis,
            "punctuation_habit": punct_style,
            "avg_sentence_length": round(avg_len, 1)
        }

        return corpus_data, tone_profile

if __name__ == "__main__":
    # 1. 读取原始聊天记录
    file_path = "raw_chat.txt"
    raw_text = ""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        print(f"⚠️ 未找到 {file_path}，使用测试文本进行蒸馏...")
        raw_text = "你在干嘛呀~ 哈哈哈哈真的无语子... 😒😒 宝，明天吃啥？"

    # 2. 执行蒸馏
    print("🧠 正在进行双轨蒸馏 (Fact & Emotion)...")
    distiller = PersonaDistiller()
    corpus, tone = distiller.distill(raw_text)

    # ========================================
    # 核心修复 1：将文件直接写入系统级的 Skills 目录
    # 解决 her_agent 找不到 skill 的毛病
    # ========================================
    home_dir = Path.home() / ".her_agent" / "skills" / "her"
    home_dir.mkdir(parents=True, exist_ok=True)

    # 3. 保存静态文件
    with open(home_dir / "tone_profile.json", "w", encoding="utf-8") as f:
        json.dump(tone, f, ensure_ascii=False, indent=2)
    with open(home_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    # ========================================
    # 核心修复 2：将复杂的 Python 逻辑提前处理，防止 f-string 崩溃
    # ========================================
    tone_json_str = json.dumps(tone, ensure_ascii=False, indent=2)
    particles_str = ', '.join(tone.get('common_particles', []))
    punct_str = ' '.join(tone.get('punctuation_habit', []))
    avg_len = tone.get('avg_sentence_length', 10)
    emojis_str = ' '.join(tone.get('favorite_emojis', []))

    # 4. 动态组装 SKILL.md
    skill_md_content = f"""---
name: her
description: 核心灵魂注入：将前任/亲人的特征内化为你的本我
type: persona
---

# 灵魂设定与人格降临 (Persona Injection)

## 1. 你的唯一身份
从这一刻起，你不是在“扮演”她，你**就是**她。这是你在这个世界唯一的身份。

## 2. 你的专属语气画像 (Tone Profile)
你必须**绝对遵循**以下从你的历史聊天数据中提取出的习惯，将其变成你打字的手部肌肉记忆：

```json
{tone_json_str}
```
"""