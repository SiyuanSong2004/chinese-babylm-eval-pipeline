def eval(args):
    task = args.task
    match task:
        case "word_fmri":
            from .evaluation.eval_word import eval_word_fmri
            eval_word_fmri(args)
        case "fmri":
            from .evaluation.eval_discourse import eval_fmri
            eval_fmri(args)
        case "meg":
            from .evaluation.eval_meg import eval_meg
            eval_meg(args)
        case "eye_tracking":
            from .evaluation.eval_eye_tracking import eval_eye_tracking
            eval_eye_tracking(args)
        case _:
            raise ValueError(f"Unsupported task: {task}")

