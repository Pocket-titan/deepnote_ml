import pandas as pd
import os

dirpath = os.path.dirname(os.path.realpath(__file__))


def make_df(path: str) -> pd.DataFrame:
    filepath = os.path.join(dirpath, path)
    data = pd.read_csv(filepath, sep=",")
    df = pd.pivot_table(
        data,  index=["Item"], columns=["Attribute"], values="TRUE", fill_value=0
    ).astype(float)
    return df


# Simple dataset
simple_df = make_df("./Rumelhart_livingthings.csv")

simple_columns = ["Grow", "Living", "LivingThing", "Animal", "Move", "Skin", "Bird", "Feathers", "Fly", "Wings", "Fish", "Gills", "Scales", "Swim", "Yellow", "Red", "Sing", "Robin",
                  "Canary", "Sunfish", "Salmon", "Daisy", "Rose", "Oak", "Pine", "Green", "Bark", "Big", "Tree", "Branches", "Pretty", "Petals", "Flower", "Leaves", "Roots", "Plant", ]
simple_index = ["Robin", "Canary", "Sunfish",
                "Salmon", "Daisy", "Rose", "Oak", "Pine"]
# Drop some columns & indices to simplify our dataset
simple_df = (
    simple_df
    .reindex(simple_index, axis="index",)
    .reindex(simple_columns, axis="columns",)
)
# In the 2020 paper, roses have no leaves; only petals (relevant for orthogonality)
simple_df["Leaves"]["Rose"] = 0.0

# Let's use a subset of the Rumelhart set for simplicity
simple_df = (
    simple_df.drop(index=["Daisy", "Pine", "Robin", "Sunfish"])
    .drop(
        columns=list(
            filter(
                lambda x: x
                not in ["Grow", "Move", "Roots", "Fly", "Swim", "Leaves", "Petals"],
                simple_columns,
            )
        )
    )
    .reindex(["Canary", "Salmon", "Oak", "Rose"], axis="index")
    .reindex(["Grow", "Move", "Roots", "Fly", "Swim", "Leaves", "Petals"], axis="columns")
)

# Extended dataset
extended_df = make_df("./Rumelhart_extended_livingthings.csv")
