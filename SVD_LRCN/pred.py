from UNet import split_vocal
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Reshape
import numpy as np
import soundfile
from Feature import AudioFeat
from sklearn.preprocessing import StandardScaler
from LRCN import get_step_data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

def preprocess(fname):
    feat_list = ['feat_spec_all', 'feat_mfcc', 'feat_MFCC_delta', 'feat_MFCC_delta_delta', 'feat_lpc', 'feat_plp']
    step = 25

    split_vocal(fname+'.wav')
    signal, samplerate = soundfile.read(fname+'.unet.Vocal.wav')
    os.remove(fname+'.unet.Vocal.wav')
    win_size_num = int(1 * samplerate) 
    hop_size_num = int(0.04 * samplerate)

    audio_feater = AudioFeat(win_size_n=win_size_num, hop_size_n=hop_size_num)
    af = audio_feater.get_audio_features(signal,samplerate)
    af = audio_feater.specify_feature_list_index(af, feat_list)

    scaler = StandardScaler()
    scaler.fit(af)
    af = scaler.transform(af)
    
    feature_dimension = np.shape(af)[-1]
    data = []
    for item_y in range(0, len(af), step):
        if item_y + step <= len(af):
            data.append(af[item_y:item_y + step])
        else:
            data.append(np.concatenate([af[item_y:], np.zeros((item_y + step - len(af), af.shape[1]))]))
    data = np.array(data)
    return data

def predict(data):
    use_model = 'standard'
    model = load_model('models/%s.model' % use_model)
    model.load_weights('models/%s.weights' % use_model)
    predict = model.predict(data)
    predict = np.where(predict > 0.5, 1, 0)
    return predict

if __name__ == '__main__':

    fname = '猫叉Master feat三澤秋 - chrono diver -fragment- (かめりあs ”crossroads of chrono” remix)'

    data = preprocess(fname)

    result = predict(data)
    f = open('pre.txt','w')
    for i in result:
        f.write(str(i[0]))