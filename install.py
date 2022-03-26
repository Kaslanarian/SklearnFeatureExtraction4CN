import os
import site


def detect_sklearn():
    for path in site.getsitepackages():
        if "sklearn" in os.listdir(path):
            print("sklearn is in %s" % path)
            return path
    else:
        print(
            "sklearn is not found, try \"pip install sklearn\" or \"pip install sklearn\""
        )
    return


def add_chinese_stopwords(stop_words_file):
    with open(stop_words_file, "a+", encoding="UTF-8") as f1:
        with open("_stop_words.py", "r", encoding="UTF-8") as f2:
            chinese_stopwords = "\n" + f2.read()
        f1.write(chinese_stopwords)


def create_cn_text(working_path):
    with open(working_path + "cn_text.py", "w", encoding="UTF-8") as f1:
        with open("text.py", "r", encoding="UTF-8") as f2:
            text = f2.read()
        f1.write(
            text.replace("sklearn", ".").replace("from _stop_words",
                                                 "from ._stop_words"))


def update_init(working_path):
    with open(working_path + "__init__.py", "r", encoding="UTF-8") as f:
        s = f.read().replace("]", ", \"cn_text\"]")
    with open(working_path + "__init__.py", "w", encoding="UTF-8") as f:
        f.write(s)


def install():
    path = detect_sklearn()
    if path != None:
        working_path = os.sep.join([
            path,
            "sklearn",
            "feature_extraction",
        ]) + os.sep

        # 增加停用词表
        add_chinese_stopwords(working_path + "_stop_words.py")
        # 加入支持中文的文本预处理
        create_cn_text(working_path)
        # 修改__init__.py，支持外部引入
        update_init(working_path)


if __name__ == "__main__":
    install()