from ml_collections.config_dict import ConfigDict


def code(cfgcase: ConfigDict):
    if cfgcase._dataloader == '2dtriangle':
        sdata = 't2'
    else:
        sdata = 'zz'
    
    if cfgcase._observe == 'grid':
        sob = 'gd'
    else:
        sob = 'zz'
    
    if cfgcase._select_model == 'ffcnn':
        smdl = 'fc'
    else:
        smdl = 'zz'

    if cfgcase._loss_fn == 'physicswithdata':
        sloss = 'pi3'
    elif cfgcase._loss_fn == 'physicsnoreplace':
        sloss = 'pi1'
    else:
        sloss = 'zz'

    s = sdata + sob + smdl + sloss

    return s

