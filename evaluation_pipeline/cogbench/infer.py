from inference.infer_sentence import infer_sentence
from inference.infer_word import infer_word

def infer(task, model_path_or_name, datapath):
    match task:
        case "word_fmri":
            return infer_word(model_path_or_name, datapath)
        case "fmri" | "meg":
            return infer_sentence(model_path_or_name, datapath)
        case _:
            raise ValueError(f"Unsupported task: {task}")

if __name__ == "__main__":
    infer("meg", "bert-base-uncased", "/mnt/backup/zhiheng2/chinese-babylm-2026/evaluation_data/cogbench/")