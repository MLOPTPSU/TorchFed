_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='emnist',
        root="./data/emnist", 
        download=False, 
        only_digits=True,
        num_classes=10,
        dimension=[28,28],
    )
)