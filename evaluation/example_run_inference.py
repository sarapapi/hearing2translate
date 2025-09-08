from util import Evaluator


# simple code to test 
metrics_to_compute = {  
                        'bleu':False, 
                        'bleurt':False, 
                        'comet':False, 
                        'comet_kiwi':True, 
                        'xcomet':False,
                        'xcomet_qe':False,
                        'metricx':False,
                        'metricx_qe':False,

                    }


evaluator = Evaluator('/gpfs/scratch/bsc88/bsc088392/hearing2translate/manifests/winoST/en-it.jsonl',
                      '/gpfs/scratch/bsc88/bsc088392/hearing2translate/outputs/seamlessm4t/winoST/en-it.jsonl',
                      'seamlessm4t')

results = evaluator.run_evaluations(metrics_to_compute)

print(results)

#[
#    {'dataset_id': 'WinoST', 'sample_id': 3888, 'src_lang': 'en', 'tgt_lang': 'it', 'output': 'La segretaria ha chiesto a qualcuno di firmare, così che possano essere emessi un guest badge.', 'metrics': {'comet_kiwi_score': 0.7255682945251465}}
#]