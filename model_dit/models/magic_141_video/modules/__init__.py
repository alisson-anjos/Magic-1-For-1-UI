from .models import Magic141VideoDiffusionTransformer, MAGIC_141_VIDEO_CONFIG

def load_model(args, in_channels, out_channels, factor_kwargs):
    """load Magic141 video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The Magic141 video model
    """
    if args.model in MAGIC_141_VIDEO_CONFIG.keys():
        model = Magic141VideoDiffusionTransformer(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **MAGIC_141_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model
    else:
        raise NotImplementedError()
