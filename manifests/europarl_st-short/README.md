
# How to use Europarl-ST

# Task
- Primary: General Benchmarking (Parlamentary intervantions)
- Secondary: ...

Can be used for both as short form (sentence level) and long form (document/intervention) dataset. The average intervention is around ~1:40 minutes long, varying across language direction. 


# Dependencies
No aditional dependencies are needed

# Description
The `generate.py` script asssumkes to be executed after preparing the long-form dataset. Set ```EUROPARL_ST_PATH``` to the
root folder of the Europarl-ST dataset, which should have been downloaded 

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