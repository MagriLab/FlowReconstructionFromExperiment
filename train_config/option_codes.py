from ml_collections.config_dict import ConfigDict


def code(cfgcase: ConfigDict):
    if cfgcase._case_dataloader == '2dtriangle':
        sdata = 't2'
    elif cfgcase._case_dataloader == '2dkol':
        sdata = 'k2'
    else:
        sdata = 'zz'
    
    if cfgcase._case_observe == 'grid':
        sob = 'gd'
    elif cfgcase._case_observe == 'grid_pin':
        sob = 'gp'
    elif cfgcase._case_observe == 'sparse':
        sob = 'sa'
    elif cfgcase._case_observe == 'sparse_pin':
        sob = 'sp'
    else:
        sob = 'zz'
    
    if cfgcase._case_select_model == 'ffcnn':
        smdl = 'fc'
    else:
        smdl = 'zz'

    if cfgcase._case_loss_fn == 'physicswithdata':
        sloss = 'pi3'
    elif cfgcase._case_loss_fn == 'physicsnoreplace':
        sloss = 'pi1'
    elif cfgcase._case_loss_fn == 'physicsreplacemean':
        sloss = 'pm3'
    elif cfgcase._case_loss_fn == 'physicsandmean':
        sloss = 'pm1'
    else:
        sloss = 'zz'

    s = sdata + sob + smdl + sloss

    return s

