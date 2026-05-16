from datetime import datetime

import numpy as np
import pandas as pd


class WandBWriter:
    def __init__(
        self,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        try:
            import wandb

            wandb.login()

            self.run_id = run_id

            wandb.init(
                project=project_name,
                entity=entity,
                config=None,
                name=run_name,
                resume="allow",
                id=self.run_id,
                mode=mode,
                save_code=kwargs.get("save_code", False),
            )
            self.wandb = wandb

        except ImportError:
            print("For use wandb install it via \n\t pip install wandb")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        return f"{object_name}_{self.mode}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_table(self, table_name, table: pd.DataFrame):
        self.wandb.log(
            {self._object_name(table_name): self.wandb.Table(dataframe=table)},
            step=self.step,
        )

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()

    def finish(self):
        self.wandb.finish()
