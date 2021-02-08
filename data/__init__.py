from .data import simple_df, extended_df
from .dataloader import make_dataloader

df = simple_df
# df = extended_df

dataloader = make_dataloader(df)
