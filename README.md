# Segmentation
Image segmentation challenge for Digital Image Processing @PKU, 2021.

File layout:
```
| - src
    | - augment.py
    | - model.py
    | - train.py
    ...
| - data
    | - complex
        | - train
            | - (images)
        | - val
            | - (images)
        | - label
            | - (labels whose filenames are the same as corresponding images,
                 no sub-folders here.)
        | - test
            | - (images)
    | - simple
        | - train
        | - val
        | - label
| - log
    | - temporary files for tensorboard
| - checkpoints
    | - *.pth (saved weights)
| - patchs
    | - crop1024 (cropped to 1024x1024)
        | - train
            | - 0189_1_1565791505_73
                | - 0_1024_0_1024.jpg (gt)
                | - 0_1024_0_1024.png (input images)
```