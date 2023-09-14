import pandas as pd
import random

def extract_just_likert_comparative(external_annotations, interactive_annotations):
    likert_external_annotations = external_annotations.xs('likert dialogue', level=1, drop_level=False)
    comparative_external_annotations = external_annotations.xs('comparative', level=1, drop_level=False)
    focused_external_annotations = pd.concat([likert_external_annotations, comparative_external_annotations])
    focused_external_annotations.columns = ['external']

    likert_interactive_annotations = interactive_annotations.xs('likert dialogue', level=1, drop_level=False)
    comparative_interactive_annotations = interactive_annotations.xs('comparative', level=1, drop_level=False)
    focused_interactive_annotations = pd.concat([likert_interactive_annotations, comparative_interactive_annotations])
    focused_interactive_annotations.columns = ['interactive']

    external_and_interactive = pd.concat([focused_external_annotations, focused_interactive_annotations],
                                         axis=1).dropna()
    return external_and_interactive

def downsample_dials(x: pd.DataFrame, n, replace=False):
    if n is None:
        return x
    return x.sample(n=n, replace=replace)
    # dialfromitem = lambda item: item[0] if isinstance(item, tuple) else item
    # dials = list({dialfromitem(indx[-1]) for indx in x.index.values})
    # sample = set(random.sample(dials, n))
    # sample_df = x[[dialfromitem(indx[-1]) in sample for indx in x.index.values]]
    # # print(f'Downsampled DF of {next(iter(x.index.values))} to {len(sample_df)} samples.')
    # return sample_df