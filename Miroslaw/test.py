import models, utils, datasets, predict
import logging, sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scales = {'Chicago': (0.78, 0.88, 1),
          'Chicago RAC': (0.6, 0.8, 0.1),
          'New Haven': (0.89, 1, 0.68),
          'New Haven RAC': (1, 1, 1),
          'Oakland': (0.84, 1, 0.51),
          'Oakland RAC': (1, 0.94, 0.23),
          'Richmond': (0.64, 1, 1),
          'Richmond RAC': (1, 1, 1)}
          
if __name__=='__main__':
    if len(sys.argv) == 1:
        print "No model name was given! Run again using format: \n\t",
        print "python test.py modelname"
    else:
        modelname = sys.argv[1]
        pred = models.test_model(modelname)
        categories = datasets.load_dataset('Categories')
        n_pred = pred.shape[0]
        pred = predict.apply_scales(pred, categories[-n_pred:], scales)
        name = modelname + ".csv"
        utils.create_submission(name, pred)
        print "Saved submission with name %s" %(name)
