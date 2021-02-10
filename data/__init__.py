from .data import simple_df, extended_df
from .dataloader import make_dataloader

simple_dataloader = make_dataloader(simple_df)
extended_dataloader = make_dataloader(extended_df)
