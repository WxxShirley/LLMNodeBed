import os 
import torch 


def save_checkpoint(model, cur_epoch, folder_str, config_str, is_best=False):
    os.makedirs(folder_str, exist_ok=True)

    param_grad_dict = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }

    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dict and not param_grad_dict[k]:
            del state_dict[k]
    
    save_obj = {
        "model": state_dict,
        "epoch": cur_epoch
    }

    path = f"{folder_str}/{config_str}_{'best' if is_best else cur_epoch}.pth"
    print("Saving checkpoint at epoch {} to {}".format(cur_epoch, path))
    torch.save(save_obj, path)


def reload_best_model(model, folder_str, config_str):
    checkpoint_path = f"{folder_str}/{config_str}_best.pth"
    
    print("Loading checkpoint from {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model 


def reload_model(model, checkpoint_path):
    print("Loading checkpoint from {}".format(checkpoint_path))
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    
    return model 
