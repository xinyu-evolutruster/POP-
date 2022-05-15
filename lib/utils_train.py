def adjust_loss_weights(init_weight, 
                        current_epoch, 
                        mode="decay",
                        pcd_start=50,
                        normal_start=200,
                        m2s_start=100,
                        s2m_start=50,
                        start=100, 
                        every=20):
    # decay or rise the loss weights according to the given policy and current epoch
    # mode: decay, rise or binary

    if mode == "normal":
        if current_epoch < normal_start:
            weight = init_weight * 1e-6
        else:
            weight = init_weight * 10.0 * (1.05 ** ((current_epoch - normal_start) // every))
    elif mode == "s2m":
        if current_epoch < s2m_start:
            weight = init_weight * 5.0
        else:
            weight = init_weight * 5.0
    elif mode == "m2s":
        if current_epoch < m2s_start:
            weight = init_weight
        else:
            weight = init_weight * 5.0
    else:
        if current_epoch < start:
            weight = init_weight
        else:
            weight = init_weight * (0.85 ** ((current_epoch - start) // every))

    """
    if mode != "binary":
        if current_epoch < start:
            if mode == "rise":
                weight = init_weight * 1e-6
            else:
                weight = init_weight
        else:
            if every == 0:
                weight = init_weight
            else:
                if mode == "rise":
                    weight = init_weight * (1.05 ** ((current_epoch - start) // every))
                else:
                    weight = init_weight * (0.85 ** ((current_epoch - start) // every))
    """
    return weight