import os

from .inference.infer_sentence import infer_sentence
from .inference.infer_word import infer_word
from .inference.infer_eye_tracking import infer_eye_tracking

def infer(args):
    """
    forward inference to get the features
    """
    task = args.task
    model_path_or_name = args.model_path_or_name
    datapath = args.data_path
    output_root = str(args.output_dir)

    match task:
        case "word_fmri":
            return infer_word(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                output_root=output_root,
                save_predictions=args.save_predictions,
                revision_name=args.revision_name,
            )
        case "fmri" | "meg":
            return infer_sentence(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                output_dir=os.path.join(output_root, os.path.basename(os.path.normpath(model_path_or_name))),
                save_predictions=args.save_predictions,
                revision_name=args.revision_name,
            )
        case "eye_tracking":
            return infer_eye_tracking(
                model_path_or_name=model_path_or_name,
                datapath=datapath,
                output_dir=os.path.join(output_root, os.path.basename(os.path.normpath(model_path_or_name))),
                save_predictions=args.save_predictions,
                revision_name=args.revision_name,
                fast=args.fast,
            )
        case _:
            raise ValueError(f"Unsupported task: {task}")
