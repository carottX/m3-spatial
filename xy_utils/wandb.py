import os
import wandb
# key: c40efa7f20dc73bee12b5ee885ea48b98f8caff4
def init_wandb(args, job_dir, entity='xuzm0616-nanjing-university', project='mmm', job_name='tmp'):
    wandb_dir = os.path.join(job_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    runid = None
    if os.path.exists(f"{wandb_dir}/runid.txt"):
        runid = open(f"{wandb_dir}/runid.txt").read()

    wandb.init(project=project,
            name=job_name,
            dir=wandb_dir,
            entity=entity,
            resume="allow",
            id=runid,
            config={"hierarchical": True},)

    open(f"{wandb_dir}/runid.txt", 'w').write(wandb.run.id)
    wandb.config.update({k: v for k, v in vars(args).items() if k not in wandb.config})