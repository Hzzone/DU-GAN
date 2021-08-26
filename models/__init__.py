from .basic_template import TrainTask
from models.DUGAN.DUGAN import DUGAN
from models.REDCNN.REDCNN import REDCNN

model_dict = {
    'DUGAN': DUGAN,
    'REDCNN': REDCNN
}
