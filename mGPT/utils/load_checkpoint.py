import torch

def load_pretrained(cfg, model, logger=None, phase="train"):
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS

    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
        
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_vae(cfg, model, logger=None):

    if isinstance(cfg.TRAIN.PRETRAINED_VAE, str):
    
        state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                                map_location="cpu", weights_only=False)['state_dict']
        if logger is not None:
            logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
            
        # Extract encoder/decoder
        from collections import OrderedDict
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if "motion_vae" in k:
                name = k.replace("motion_vae.", "")
                vae_dict[name] = v
            elif "vae" in k:
                name = k.replace("vae.", "")
                vae_dict[name] = v

        if hasattr(model, 'vae'):
            model.vae.load_state_dict(vae_dict, strict=True)
        else:
            model.motion_vae.load_state_dict(vae_dict, strict=True)

    elif "dict" in  str(type(cfg.TRAIN.PRETRAINED_VAE)):
        for key, ckpt in cfg.TRAIN.PRETRAINED_VAE.items():
            modality_specific_model = getattr(model.vae, key)

            state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)['state_dict']

            # Extract encoder/decoder
            from collections import OrderedDict
            vae_dict = OrderedDict()
            for k, v in state_dict.items():
                if "motion_vae" in k:
                    name = k.replace("motion_vae.", "")
                    vae_dict[name] = v
                elif "vae" in k:
                    name = k.replace("vae.", "")
                    vae_dict[name] = v

            if logger is not None:
                logger.info(f"Loading pretrain vae:{key} from {ckpt}")
            else:
                print(f"Loading pretrain vae:{key} from {ckpt}")

            modality_specific_model.load_state_dict(vae_dict, strict=True)

    
    return model
