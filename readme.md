N.B.: This project incorporates code from [BinaryNet.pytorch](https://github.com/itayhubara/BinaryNet.pytorch) by Itay Hubara. The author's code is licensed under the MIT license and is open to use in public. The following illustrates the functions of each file.


- **software**: Root Directory
    - **project_code.ipynb**: this file contains the code implemented for the software part.
    - **binarized_modules**: this file contains the class for binarization modules.
    - **converted weights**: this file contains the codes to convert the weights.

    - **txt file**: 
        - **example.txt**: The flattened and binarized MNIST data will be stored in this file.
        - **label compare.txt**: This file store the target label and predicted label. The first figure is the true label, the second figure is the predicted label (e.g., 42, 4 is the true label, 2 is the predicted label).
        - **before_softmax_output.txt**: This file stores the output before softmax, which helps to debug.
    - **readme.md**
