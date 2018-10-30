import os, sys
sys.path.append(".")

from maqc.models.trainer import ModelTrainer
import numpy as np

def predict_quality(X, platform = None, script_dir = "."):
    """Use pretrained model to give estimate of quality

    The model is platform dependent and you can pass in full set of feaures. They
    will be subset to indices retained on training.

    Parameters:
    ------------
    X: np.ndarray
        samples x features matrix
    platform: str
        name of platform
    script_dir: str
        path to maQCN_pipeline folder
    """
    if platform is None: platform = "Affymetrix"
    if platform.lower() in ("affymetrix", ):
        fpath = os.path.join(script_dir, "ml/estimators/RandomizedSearchCV_10261306.pkl")
    else:
        msg = "Platform {} is not implemented".format(platform)
        raise NotImplementedError(msg)
    mt = ModelTrainer(rnd = 56)
    mt.load_model(fpath)
    y_hat = mt.predict(X, proba = True)

    return(y_hat)


# if __name__ == "__main__":
#     X = np.random.normal(size = (5, 133))
#     y_hat = predict_quality(X)
#     print(y_hat)
