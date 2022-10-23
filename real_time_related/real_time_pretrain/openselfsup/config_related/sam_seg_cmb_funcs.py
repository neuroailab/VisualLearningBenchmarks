from . import saycam_funcs, gnrl_funcs, simclr_cfg_funcs


seg0to50_cfg_func = gnrl_funcs.sequential_func(
        saycam_funcs.sam_seg_range_ep300_cfg_func(0, 50),
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        )

seg50to100_cfg_func = gnrl_funcs.sequential_func(
        saycam_funcs.sam_seg_range_ep300_cfg_func(50, 100),
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        )

seg0to100_cfg_func = gnrl_funcs.sequential_func(
        saycam_funcs.sam_seg_range_ep300_cfg_func(0, 100),
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        )
