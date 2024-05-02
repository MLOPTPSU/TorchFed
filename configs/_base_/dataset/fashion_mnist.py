_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='fashion_mnist',
        root="./data/fashion_mnist", 
        download=False,
        num_classes=10,
        dimension=[28,28],
        )
)