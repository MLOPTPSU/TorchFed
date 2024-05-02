_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='emnist',
        root="./data/emnist", 
        download=False, 
        only_digits=False,
        num_classes=62,
        dimension=[28,28],
    )
)