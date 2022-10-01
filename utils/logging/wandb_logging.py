
import wandb
class Wandblogger:
    def __int__(self,job_type='Training',name):
        wandb.init(job_type=job_type,
                   entity="fsdl2022_project051",
                   project='land_cover_segmentation',
                   name = name)
        self.job_type = job_type
        self.log_dict = {}
        self.wandb, self.wandb_current_run = wandb, wandb.run
    def log(self,log_dict):
        # if self.wandb_current_run:
        for key, value in log_dict.items():
            self.log_dict[key] = value
    def end_epoch(self):
        if self.wandb_current_run:
            wandb.log(self.log_dict)
            self.log_dict = {}

    def end_batch(self):
        if self.wandb_current_run:
            wandb.log(self.log_dict)
            self.log_dict = {}

def generate_run_name(self):
    from datetime import datetime
    dt = datetime.now()
    dt_str = dt.strftime('%d%b%y_%H%M%S')
    return dt_str


            