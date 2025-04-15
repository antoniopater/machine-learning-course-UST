import pickle
import pandas as pd

def save_mse(models_obj, path="outputs/mse.pkl"):
    mse_dict = models_obj.mse_models
    df = pd.DataFrame({
        'train_mse': [mse_dict[m]['train_mse'] for m in mse_dict],
        'test_mse':  [mse_dict[m]['test_mse'] for m in mse_dict]
    }, index=mse_dict.keys())
    df.to_pickle(path)
    return df

def save_models(models_list, path="outputs/reg.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(models_list, f)
