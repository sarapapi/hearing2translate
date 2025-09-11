import os
def load_model():
    # Return a dummy model for testing purposes
    return "test_model"

def generate(model, sample):
    audio = sample["sample"]
#    if not os.path.exists(audio):
#        raise FileNotFoundError(f"Audio file {audio} does not exist.")
    return f"{audio} in {sample['tgt_lang']} exists"