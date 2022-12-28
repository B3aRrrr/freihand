'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de


About this file:
================
This file defines a wrapper for the loading functions of the MANO model. 

Modules included:
- load_model:
  loads the MANO model from a given file location (i.e. a .pkl file location), 
  or a dictionary object.

'''

def ready_arguments(fname_or_dict, posekey4vposed='pose'):
    import numpy as np
    import pickle
    import chumpy as ch
    from chumpy.ch import MatVecMult
    from utils.mano_core.posemapper import posemap

    if not isinstance(fname_or_dict, dict):
      # dd = pickle.load(open(fname_or_dict)) #  OLD
      with open(fname_or_dict,'rb') as f:
        dd = pickle.load(f ,encoding='bytes')
    else:
        dd = fname_or_dict
    #print(f'dd {dd.keys()}')
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd[b'kintree_table'].shape[1]*3

    if b'trans' not in dd:
        dd[b'trans'] = np.zeros(3)
    if b'pose' not in dd:
        dd[b'pose'] = np.zeros(nposeparms)
    if b'shapedirs' in dd and b'betas' not in dd:
        print(dd.get(b'shapedirs').compute_r())
        dd[b'betas'] = np.zeros(dd[b'shapedirs'].shape[-1])

    for s in [b'v_template', b'weights', b'posedirs', b'pose', b'trans', b'shapedirs', b'betas', b'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert(posekey4vposed in dd)
    if want_shapemodel:
        dd[b'v_shaped'] = dd[b'shapedirs'].dot(dd[b'betas'])+dd[b'v_template']
        v_shaped = dd[b'v_shaped']
        J_tmpx = MatVecMult(dd[b'J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd[b'J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd[b'J_regressor'], v_shaped[:, 2])
        dd[b'J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd[b'v_posed'] = v_shaped + dd[b'posedirs'].dot(posemap(dd[b'bs_type'])(dd[posekey4vposed]))
    else:
        dd[b'v_posed'] = dd[b'v_template'] + dd[b'posedirs'].dot(posemap(dd[b'bs_type'])(dd[posekey4vposed]))

    return dd


def load_model(fname_or_dict, ncomps=6, flat_hand_mean=False, v_template=None, use_pca=True):
    ''' This model loads the fully articulable HAND SMPL model,
    and replaces the pose DOFS by ncomps from PCA'''

    from utils.mano_core.verts import verts_core
    import numpy as np
    import chumpy as ch
    import pickle
    import scipy.sparse as sp
    np.random.seed(1)

    if not isinstance(fname_or_dict, dict):
      with open(fname_or_dict,'rb') as f:
        smpl_data = pickle.load(f ,encoding='bytes')
    else:
      smpl_data = fname_or_dict

    #print(f'smpl_data {smpl_data.keys()}')
    rot = 3  # for global orientation!!!

    if use_pca:
        hands_components = smpl_data[hands_components]  # PCA components
    else:
        hands_components = np.eye(45)  # directly modify 15x3 articulation angles
    hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data[b'hands_mean']
    hands_coeffs = smpl_data[b'hands_coeffs'][:, :ncomps]

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    pose_coeffs = ch.zeros(rot + selected_components.shape[0])
    full_hand_pose = pose_coeffs[rot:(rot+ncomps)].dot(selected_components)

    smpl_data[b'fullpose'] = ch.concatenate((pose_coeffs[:rot], hands_mean + full_hand_pose))
    smpl_data[b'pose'] = pose_coeffs

    Jreg = smpl_data[b'J_regressor']
    if not sp.issparse(Jreg):
        smpl_data[b'J_regressor'] = (sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))

    # slightly modify ready_arguments to make sure that it uses the fullpose 
    # (which will NOT be pose) for the computation of posedirs
    dd = ready_arguments(smpl_data, posekey4vposed=b'fullpose')

    # create the smpl formula with the fullpose,
    # but expose the PCA coefficients as smpl.pose for compatibility
    args = {
        'pose': dd[b'fullpose'],
        'v': dd[b'v_posed'],
        'J': dd[b'J'],
        'weights': dd[b'weights'],
        'kintree_table': dd[b'kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd[b'bs_style'],
    }

    result_previous, meta = verts_core(**args)
    result = result_previous + dd[b'trans'].reshape((1, 3))
    result.no_translation = result_previous

    if meta is not None:
        for field in [b'Jtr', b'A', b'A_global', b'A_weighted']:
            if(hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    if hasattr(result, b'Jtr'):
        result.J_transformed = result.Jtr + dd[b'trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    if v_template is not None:
        result.v_template[:] = v_template
    result.dd = dd
    return result

if __name__ == '__main__':
    m = load_model()
    m.J_transformed
    print ("FINITO")
