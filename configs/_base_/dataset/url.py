_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='url',
        root="./data/url",
        num_classes=2,
        dimension=[3231961],
    )
)