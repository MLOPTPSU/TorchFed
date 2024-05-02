_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='shakespeare',
        root="./data/shakespeare", 
        download=False, 
        batch_size=2, 
        seq_len=50,
        num_classes=90,
        dimension=[90],
        )
)