def parse_configs():
    import configargparse
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = "my POP"
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog="POP")
    
    # general
    parser.add_argument("--config", is_config_file=True)
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "resume", "test", "test_seen", "test_unseen"])
    parser.add_argument("--punet", type=bool, default=False)

    # architecture
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--pose_feature_channel", type=int, default=64)
    parser.add_argument("--geo_feature_channel", type=int, default=64)
    parser.add_argument("--num_body_verts", type=int, default=10475)

    # posmap
    parser.add_argument("--query_posmap_size", type=int, default=256)
    parser.add_argument("--meanshape_posmap_size", type=int, default=128)

    # loss
    parser.add_argument("--w_m2s", type=float, default=1e4)
    parser.add_argument("--w_s2m", type=float, default=1e4)
    parser.add_argument("--w_normal", type=float, default=1.0)
    parser.add_argument("--w_lrd", type=float, default=2e3)
    parser.add_argument("--w_lrg", type=float, default=1.0)

    # training / eval related
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_geometry", type=float, default=5e-4)
    parser.add_argument("--decay_start", type=int, default=250)
    parser.add_argument("--decay_every", type=int, default=400)
    parser.add_argument("--rise_start", type=int, default=250)
    parser.add_argument("--rise_every", type=int, default=400)
    parser.add_argument("--val_every", type=int, default=1)

    args, _ = parser.parse_known_args()

    return args

def parse_outfits(exp_name):
    '''
    parse the seen/unseen outfits configuration defined in configs/clo_config.yaml
    '''
    import yaml

    clothing_config_path = "configs/clo_config.yaml"
    config_all = yaml.load(open(clothing_config_path), Loader=yaml.SafeLoader)

    config_exp = config_all[exp_name]
    return_dict = {}
    for key, value in config_exp.items():
        value = sorted(value)
        value_dict = dict(zip(value, range(0, len(value))))
        return_dict[key] = value_dict

    return return_dict