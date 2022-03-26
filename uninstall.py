import os
from install import detect_sklearn


def remove_chinese_stopwords(stop_words_file):
    with open(stop_words_file, "r", encoding="UTF-8") as f:
        s = f.read()
    with open(stop_words_file, "w", encoding="UTF-8") as f:
        index = s.find("CHINESE_STOP_WORDS")
        if index == -1:
            print("CHINESE_STOP_WORDS not found")
        f.write(s[:index])


def remove_cn_text(working_path):
    os.remove(working_path + "cn_text.py")


def repair_init(working_path):
    with open(working_path + "__init__.py", "r", encoding="UTF-8") as f:
        s = f.read().replace(", \"cn_text\"", "")
    with open(working_path + "__init__.py", "w", encoding="UTF-8") as f:
        f.write(s)


def uninstall():
    path = detect_sklearn()
    if path != None:
        working_path = os.sep.join([
            path,
            "sklearn",
            "feature_extraction",
        ]) + os.sep

        # 删除应用词表
        remove_chinese_stopwords(working_path + "_stop_words.py")
        # 删除cn_text
        remove_cn_text(working_path)
        # 还原__init__.py
        repair_init(working_path)


if __name__ == "__main__":
    uninstall()