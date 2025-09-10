
# How to use Europarl-ST

# Task
- Primary: General Benchmarking (Parlamentary intervantions)
- Secondary: ...

Can be used for both as short form (sentence level) and document level for dataset. The average intervention is around ~1:40 minutes long, varying across language direction. 
Here we only support the short form version.


# Dependencies
No aditional dependencies are needed

# Description
The `generate.py` script downloads and prepares the dataset from the original upluad for the langs supported by this repo. It needs to download the whole dataset, including training, which is around 20 GBs. Audio data is resampled to 16Hz and changed into .wav for easier processing of later models. You can set ```EUROPARL_ST_PATH``` to the root folder of the Europarl-ST dataset if you already have the dataset downloaded.

```bash
python generate.py 
```
# Languages directions
- en->{es,fr,pt,it,de}->en

# Reference and Links
https://www.mllp.upv.es/europarl-st/
https://ieeexplore.ieee.org/document/9054626
https://arxiv.org/abs/1911.03167

# License
The europarl-ST dataset was released under a NonCommercial 4.0 International license.