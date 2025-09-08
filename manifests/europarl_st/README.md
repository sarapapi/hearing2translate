
# How to use Europarl-ST

# Task
- Primary: General Benchmarking (Parlamentary intervantions)
- Secondary: 

Can be used for both as short form (sentence level) and long form (document/intervention) dataset. The average itnervantion is around ~1:40 minutes long, varying across language direction. 

# Dependencies
No aditional dependencies are needed

# Description
The `generate.py` script downloads and prepares the dataset from the original upluad for the langs supported by this repo. It needs to download the whole dataset, including training, which is around 20 GBs


```bash
python generate.py 
```
# Languages directions
- de-en
- es-en
- fr-en

# Reference and Links
https://www.mllp.upv.es/europarl-st/
https://ieeexplore.ieee.org/document/9054626
https://arxiv.org/abs/1911.03167

# License
The europarl-ST dataset was released under a NonCommercial 4.0 International license.