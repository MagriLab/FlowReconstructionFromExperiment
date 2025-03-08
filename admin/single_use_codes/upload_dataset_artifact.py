import wandb
import h5py
from pathlib import Path


run = wandb.init(project="FlowReconstruction", name="CreateArtifactDataPath")


kolsolsets = Path('./local_data/kolmogorov/dim3_datasets_080325.txt')
assert kolsolsets.exists()
tag = "$".join(kolsolsets.parts)
data = wandb.Artifact(
    '3dkolsets',
    type='DataPath',
    description="Collection of 6 sets of 3D Kolmogorov flows",
    metadata={
        're': 34,
        'filepath': str(kolsolsets),
        'random_seeds': [478,57,1372,691],
        'number_of_sets': 6,
        'snapshots_per_set': 500,
        'flowtype': "turbulent",
        "source": "KolSol",
        "notes": "Weakly turbulent. First 5 sets are used when testing for convergence (see attached figure)."
    }
)
data.add_file(Path('./local_data/kolmogorov/Kolsol_converge_080325.png'))
run.log_artifact(data, aliases=[tag])


# d1 = Path('./local_data/re100/')
# tag1 = "$".join(d1.parts)
# data1 = wandb.Artifact(
#     '2dtriangle',
#     type="DataPath",
#     description="2D wake of a triangular body at Re=100",
#     metadata={
#         're': 100,
#         'filepath': str(d1),
#         'snapshots': 800,
#         'flowtype': "laminar",
#         'source': "xcompact3d",
#         "notes": ""
#     }
# )
# data1.add_file('./local_data/re100/fig_dataset_example.png')
# data1.add_file('./local_data/re100/fig_dataset_probes.png')
# data1.add_file('./local_data/re100/fig_validate_uz.png')
# data1.add_file('./local_data/re100/input.i3d')
# run.log_artifact(data1, aliases=[tag1])


# d2 = Path('./local_data/kolmogorov/dim2_re34_k32_f4_dt1_grid128_14635.h5')
# tag2 = "$".join(d2.parts)
# data2 = wandb.Artifact(
#     '2dkol',
#     type='DataPath',
#     description="Kolmogorov flow generated with random seed 14635",
#     metadata={
#         're': 34,
#         'filepath': str(d2),
#         'random_seed': 14635,
#         'snapshots': 10000,
#         'flowtype': "turbulent",
#         "source": "KolSol",
#         "notes": "Weakly turbulent, see Alberto's thesis."
#     }
# )
# data2.add_file('fig_dissipation_14635.png')
# data2.add_file('fig_mean_14635.png')
# run.log_artifact(data2, aliases=[tag2])


# d3 = Path('./local_data/kolmogorov/dim2_re34_k32_f4_dt1_grid128_25619.h5')
# tag3 = "$".join(d3.parts)
# data3 = wandb.Artifact(
#     '2dkol',
#     type='DataPath',
#     description="Kolmogorov flow generated with random seed 25619",
#     metadata={
#         're': 34,
#         'filepath': str(d3),
#         'random_seed': 25619,
#         'snapshots': 10000,
#         'flowtype': "turbulent",
#         "source": "KolSol",
#         "notes": "Weakly turbulent, see Alberto's thesis."
#     }
# )
# data3.add_file('fig_dissipation_25619.png')
# data3.add_file('fig_mean_25619.png')
# run.log_artifact(data3, aliases=[tag3])



# d4 = Path('./local_data/kolmogorov/dim2_re40_k32_dt1_T800_grid128_586178_short.h5')
# with h5py.File(d4) as hf:
#     re = float(hf.get('re')[()])
# tag4 = "$".join(d4.parts)
# data4 = wandb.Artifact(
#     '2dkol',
#     type='DataPath',
#     description="Kolmogorov flow generated with extreme events. Generated at dt 0.005, saved at dt 0.1.",
#     metadata={
#         're': re,
#         'filepath': str(d4),
#         'random_seed': 586178,
#         'snapshots': 6500,
#         'flowtype': "turbulent",
#         "source": "KolSol",
#         "notes": "With extreme events. "
#     }
# )
# data4.add_file('fig_dissipation_over_time.png')
# data4.add_file('fig_mean_re40.png')
# run.log_artifact(data4, aliases=[tag4])



run.finish()
