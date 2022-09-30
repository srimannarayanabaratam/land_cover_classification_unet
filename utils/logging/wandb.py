
import wandb
class Wandblogger:
    def __int__(self,job_type='Training'):
        self.job_type = job_type
        self.log_dict = {}
        self.wandb, self.wandb_current_run = wandb, wandb.run
    def log(self,log_dict):
        if self.wandb_current_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value
    def end_epoch(self):
        if self.wandb_current_run:
            wandb.log(self.log_dict)
            self.log_dict = {}
            