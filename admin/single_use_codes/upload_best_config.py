import wandb
from argparse import ArgumentParser
from pathlib import Path


def main(args):

    fpath = Path(args.f)
    assert fpath.exists(), 'Cannot find config file.'
    
    wandb.init(project=args.project, id=args.id, resume="must")
    f = wandb.Artifact(
        args.name,
        type='Configs',
    )
    f.add_file(fpath)
    if args.tags:
        tags = args.tags.split(',')
    else:
        tags=None
    wandb.log_artifact(f, aliases=tags) 

    wandb.finish()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--f', required=True, help='Path to the config file')
    parser.add_argument('--project', default='FlowReconstruction', help='Name of project')
    parser.add_argument('--id', required=True, help='Run ID')
    parser.add_argument('--name', required=True, help='Name of the artifact')
    parser.add_argument('--tags', help='Tags to add to the artifact, separated with comma')
    args = parser.parse_args()
    main(args)
