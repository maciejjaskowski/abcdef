from __future__ import division
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split

class Computation(object):        
    def properties(self):
      return filter(lambda x: not x.startswith("_"), dir(self))

def as_contiguous_df(df):
  return pd.DataFrame(data = np.ascontiguousarray(df.values), 
                             index = df.index,
                             columns = df.columns)

def load_fc7(directory_out):
  import cPickle as pickle

  l = [(x[:4], x[4:-8], x[-8:]) for x in sorted(list(os.listdir(directory_out)))]
  fc7 = sorted([(int(x[1]), directory_out + x[0] + x[1] + x[2]) for x in l if x[0] == 'fc7_'])

  rs = []
  for i, f in fc7:
    with open(f, 'r') as handle:
      rs.append(pickle.load(handle))

  return np.concatenate(rs, axis = 0)


def prep_rs(rs):
    directory = './train_photos/'
    photo_ids = [int(x[4:-4]) for x in sorted(list(os.listdir(directory)))]
    photo_prefix = dict([(int(x[4:-4]),x[:4]) for x in sorted(list(os.listdir(directory)))])	
    photo_to_biz_train = pd.read_csv('train_photo_to_biz_ids.csv', header = 0, index_col = 'photo_id')

    df_rs = pd.DataFrame(rs, index = photo_ids)
    df_rs.index.rename("photo_id", inplace = True)
    df_rs_with_biz = pd.concat([df_rs, photo_to_biz_train], axis = 1, join = 'inner' )
    assert photo_to_biz_train["business_id"].ix[204149] == photo_to_biz_train.ix[204149].values[0]
    assert np.shape(photo_to_biz_train)[0] == np.shape(df_rs)[0]

    df_rs_cleaned_up = df_rs_with_biz[df_rs_with_biz['business_id'].apply(lambda x: x not in [430, 1627, 2661, 2941])]    
    return photo_to_biz_train, df_rs_cleaned_up, photo_prefix

def prep_test(rs_test):	
	directory_test = './test_photos/'
	photo_ids_test = [int(x[:-4]) for x in sorted(list(os.listdir(directory_test)))]	
	df_rs_test = pd.DataFrame(rs_test, index = photo_ids_test)

	photo_to_biz_test = pd.read_csv('test_photo_to_biz.csv', header = 0)

	df_rs_test.index.rename("photo_id", inplace = True)
	return df_rs_test


def prep_y(train):
    n_labels = 9
    # remove NaNs
    train['labels'].isnull()
    train2 = train[train['labels'].notnull()]

    train3 = train2
    train3['labels_list'] = train2['labels'].str.split().values.tolist()

    for i in range(n_labels):
      train3[str(i)] = train3['labels_list'].apply(lambda row: str(i) in row).astype(int)
  
    return train3

def test_pred_with_business_id(df_pred):
  directory_test = './test_photos/'
  photo_ids_test = [int(x[:-4]) for x in sorted(list(os.listdir(directory_test)))]
  df_pred = pd.DataFrame(pred, index = photo_ids_test)
  photo_to_biz_test = pd.read_csv('test_photo_to_biz.csv', header = 0)
  return df_pred.merge(photo_to_biz_test, left_index = True, right_on = 'photo_id')  
  
def train_validation_split(df_Xy, business_ids, ratio):
  import random
  import math
  random.seed(49812479)
  uniq_business_ids = set(business_ids)
  train_business_ids = set(random.sample(list(uniq_business_ids), int(math.ceil(len(uniq_business_ids) * 1.0 * ratio))))
  validation_business_ids = uniq_business_ids - train_business_ids

  train_indices = df_Xy['business_id'].isin(train_business_ids)
  validation_indices = df_Xy['business_id'].isin(validation_business_ids)
  df_Xy_train = df_Xy[train_indices]
  df_Xy_valid = df_Xy[validation_indices]

  return df_Xy_train, df_Xy_valid
  #train_business_ids = uniq_business_ids
  #validation_business_ids = set(photo_to_biz_test['business_id'].values)

def threshold(gb, lvl):
  business_pred = gb
  business_pred[business_pred > lvl] = 1
  business_pred[business_pred <= lvl] = 0  
  return business_pred

def predict_business_with_mean(df_pred_with_business_id):
  result_mean = df_pred_with_business_id.groupby(['business_id']).mean().iloc[:,0:9]
  result_mean.sort_index(inplace = True)
  y_business_pred = threshold(result_mean.copy(), 0.5)  
  return y_business_pred, result_mean