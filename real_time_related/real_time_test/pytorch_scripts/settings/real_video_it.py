from . import xx_basic


def add_real_video_it_setting(args):
    args.include_test = True
    args.num_steps = 1350
    args.eval_freq = 75
    args.test_more_trials = True
    args.batch_size = 32
    args.real_video_aggre_time = 0.2
    args.concat_batch = True
    return args

def add_func(_func):
    all_things = globals()
    def new_func(args):
        args = getattr(xx_basic, _func)(args)
        args = add_real_video_it_setting(args)
        return args
    if _func not in all_things:
        all_things[_func] = new_func

all_basic_funcs = dir(xx_basic)
for _func in all_basic_funcs:
    if _func.endswith('mix_break')\
            or _func.endswith('mix_build')\
            or _func.endswith('mix_switch'):
        add_func(_func)
