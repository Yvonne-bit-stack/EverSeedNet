def get_model(model_name, args):
    name = model_name.lower()
    if name == "ever_seed_net":
        from models.ever_seed_net import EverSeedNet
        return EverSeedNet(args)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Only 'ever_seed_net' is supported.")
