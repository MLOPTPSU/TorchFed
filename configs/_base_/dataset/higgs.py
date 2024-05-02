_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='higgs',
        root="./data/higgs", 
        num_classes=2,
        dimension=[28],
    )
)