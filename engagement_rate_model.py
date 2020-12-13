# Import libraries
import numpy as np
from numpy import asarray 
import pandas as pd
import requests
from urllib.request import urlopen
import csv

import glob
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from datetime import date 

import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import GaussianNoise, Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation, BatchNormalization, Dropout, concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn import tree

from tqdm import tqdm
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')

import textblob
import re
from textblob import TextBlob
from bs4 import BeautifulSoup
import html
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

# Parse vehicles to read in
file = 'data_09092020/vehicles_20200909.csv' # Change file variable to absolute path of vehicles.csv on your machine
col_names = ['vehicle_id', 'year', 'make', 'model', 'trim', 'body', 'stock_number', 'vin', 'used', 'certified', 'exterior_color',
'interior_color', 'doors', 'mileage', 'engine', 'drivetrain', 'price', 'msrp', 'sale_price', 'description', 'date_first_on_lot',
'vdp_url', 'dealer_id', 'created_at', 'deleted_at']
df_list = []

with open(file) as csv_file:
    reader = csv.reader(csv_file, quotechar='"', delimiter=',', escapechar='\\')
    for row in reader:
        if len(row) == 25:
            for i in range(len(row)):
                 if '\n' in row[i]:
                        row[i] = row[i].replace("\n", '')
            df_list.append(row)

vehicles_df = pd.DataFrame(columns=col_names, data=df_list)

"""
If running this in the future, please utilize this code from performance.csv to scrape images from the web. 
If running on current data, skip because jpg images have already been saved

performance_df = pd.read_csv('performance.csv')
for i in range(len(performance_df.image)):
    print(i)
    url = performance_df.image[i]
    if type(performance_df.image[i]) == str and performance_df.image[i][0:4] == 'http':
        try:
            r = requests.get(url)
            if r.status_code == 404:
                continue
            else:
                image_name = "img_" + str(performance_df.veh_id[i]) + "_.jpg"
                with open(image_name, 'wb') as f:
                    f.write(r.content)
        except requests.exceptions.ConnectionError:
            continue

"""

"""
Read in jpg images from file folder on local machine. 
If images cannot be reshaped to [200, 200, 3], they are most likely not RGB and will be discarded
Append vehicle id to a list and write it to csv for future filtering

working_dir = '/home/oroberts/DealersUtd/jpg_images/*.jpg' # your working directory here
jpg_imgs = glob.glob(working_dir)
id_array = []
for i in range(len(jpg_imgs)):
    try:
        img = np.array(Image.open(jpg_imgs[i]).resize((200,200)))
        img = img.reshape([200, 200, 3])
        id_array.append(jpg_imgs[i].split('_')[2])
        img.close()
    except:
        continue

with open('id_images.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerow(id_array)

"""

"""
Merging 'performance' and 'vehicles' to get the column description added to performance
"""
# Loading 'performance' and changing the names of columns we need to merge on
performance_df = pd.read_csv('performance.csv')
performance_df = performance_df.rename({'id':'veh_id'}, axis='columns')

# Loading 'vehicles' and changing the names of columns we need to merge on
vehicles_df = vehicles_df.rename({'vehicle_id':'veh_id'}, axis='columns')

# Dropping rows that have random things in dealer_id aka dealer_id is not an integer
a = vehicles_df['dealer_id'].str.isnumeric()
subset_vehicles_df = vehicles_df[a]

# Changing types to prepare for merging
performance_df['veh_id']=performance_df['veh_id'].astype(int)
subset_vehicles_df['veh_id']=subset_vehicles_df['veh_id'].astype(int)
performance_df['dealer_id']=performance_df['dealer_id'].astype(int)
subset_vehicles_df['dealer_id']=subset_vehicles_df['dealer_id'].astype(int)

# Merging performance and vehicles
merged_perf_veh = pd.merge(performance_df, subset_vehicles_df, on=['veh_id','dealer_id'])

# Dropping double columns and renaming some columns to original name
merged_perf_veh = merged_perf_veh.drop(['year_y', 'make_y', 'model_y', 'trim_y',  'body','stock_number_y', 'vin_y','used_y',
                     'certified_y','exterior_color','interior_color','doors','mileage','engine','drivetrain','price',
                     'msrp','sale_price','date_first_on_lot','vdp_url','created_at_y','deleted_at_y'], axis = 1)
merged_perf_veh = merged_perf_veh.rename({'vin_x':'vin', 'make_x':'make', 'model_x':'model','trim_x':'trim',
                        'stock_number_x':'stock_number','used_x': 'used','certified_x':'certified','year_x':'year',
                        'created_at_x':'created_at', 'deleted_at_x':'deleted_at'}, axis='columns')

# Checking nulls 
merged_perf_veh.isnull().sum()

# Encoding 'NULL' as NaN and checking again
merged_df = merged_perf_veh.replace('NULL', np.NaN)
merged_df.isnull().sum()

"""
In this section the description column will be cleaned
"""

# Stripping html tags from the description
def clean_text(text):
    text = BeautifulSoup(html.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

clean = []
for i in range(len(merged_df)):
    if type(merged_df['description'].iloc[i]) != type(None):
        merged_df.description.iloc[i] = clean_text(str(merged_df['description'].iloc[i]))
    else:
        pass

nltk.download('stopwords')
stop = stopwords.words('english')

# Removing stop words and saving it to a new column so we can extract features from
merged_df['description_without_stopwords'] = merged_df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

"""
This section contains feature engineering
"""

# Adding column that give us the total number of words in the description (excludes stopwords)
merged_df['totalwords'] = merged_df['description_without_stopwords'].str.split().str.len()

# Adding column to get ratio of upper to lower case letters in the description
merged_df['uppercase_to_total_words'] = (merged_df['description_without_stopwords'].str.findall(r'[A-Z]').str.len())/(merged_df['totalwords'])

# Write it to a csv in the future...
#merged_df.to_csv('performance_description.csv')

"""
Merged_df is now 'performance_description'
"""

# Reading in dataset
performance_description = pd.read_csv('performance_description.csv', index_col=0)

# Defining functions to calculate sentiment score and subjectivity score
def sentiment(x):
    sentiment = TextBlob(str(x))
    return sentiment.sentiment.polarity

def subjectivities(x):
    sentiment = TextBlob(str(x))
    return sentiment.sentiment.subjectivity

# Defining functions to calculate sentiment score and subjectivity score
def sentiment(x):
    sentiment = TextBlob(str(x))
    return sentiment.sentiment.polarity


def subjectivities(x):
    sentiment = TextBlob(str(x))
    return sentiment.sentiment.subjectivity

# Adding columns for sentiment ('sentiment_score') and subjectivity ('subjectivity_score')
performance_description['sentiment_score'] = performance_description['description'].apply(sentiment)
performance_description['subjectivity_score'] = performance_description['description'].apply(subjectivities)

# Adding column ('numeric') that tells how many numbers are in each text
performance_description['numeric'] = performance_description['description'].astype(str).apply(lambda x : len([x for x in x.split() if x.isdigit()]))

# If description is NaN then make the columns "totalwords","uppercase_to_total_words", "sentiment_score",	"subjectivity_score",	"numeric" also NaN
for i in range(len(performance_description)):
  if pd.isna(performance_description['description'].iloc[i]) == True:
    performance_description["totalwords"].iloc[i] = np.NaN
    performance_description["uppercase_to_total_words"].iloc[i] = np.NaN
    performance_description["sentiment_score"].iloc[i] = np.NaN
    performance_description["subjectivity_score"].iloc[i] = np.NaN
    performance_description["numeric"].iloc[i] = np.NaN
  else:
    pass

"""
Now the dataset 'performance_description', which contains engineered features, will be merged with a filtered dataset that only 
contains vehicles we have images for
"""

#filtered_df.csv can be obtained from the id_images.csv (line 103-105)
id_df = pd.read_csv('id_images.csv', header = None).T
id_df.columns = ['id']
id_array = id_df.id.tolist()
filtered_df = performance_df.loc[performance_df['veh_id'].isin(id_array)]
filtered_df = filtered_df.drop_duplicates(['veh_id'], keep='last')
#filtered_df.to_csv('filtered_df.csv')

# Reading in the filtered df which contains only vehicles we haves images for
filt_perf = pd.read_csv('filtered_df.csv', index_col=0)
filt_perf = filt_perf.rename({'id':'veh_id'}, axis='columns')

# Left merging on vehicle id, vin, and dealer_id 
merged = pd.merge(filt_perf, performance_description, how='left', on=['veh_id', 'vin', 'dealer_id'])

# Removing duplicates
merged_new = merged.drop_duplicates()

# Dropping double columns
merged_new1 = merged_new.drop(['spend_y', 'lead_y', 'impressions_y', 'reach_y', 'traffic_y',
       'engagement_y', 'content_views_y', 'website_leads_y', 'make_y',
       'model_y', 'trim_y', 'stock_number_y', 'used_y', 'certified_y',
       'year_y', 'created_at_y', 'deleted_at_y', 'sold_y', 'image_y',
       'day_created_at_y', 'number_days_posted_y'], axis=1)

# Changing column names back to how they originally were
perform_features = merged_new1.rename({'spend_x':'spend', 'lead_x':'lead', 'impressions_x':'impressions',
       'reach_x':'reach', 'traffic_x':'traffic', 'engagement_x':'engagement', 'content_views_x':'content_views',
       'website_leads_x':'website_leads', 'make_x':'make', 'model_x':'model', 'trim_x':'trim', 'stock_number_x':'stock_number',
       'used_x':'used', 'certified_x':'certified', 'year_x':'year', 'created_at_x':'created_at', 'deleted_at_x':'deleted_at',
       'sold_x':'sold', 'image_x':'image', 'day_created_at_x':'day_created_at', 'number_days_posted_x':'number_days_posted'}, axis='columns')

# Write it to a csv in the future...
#perform_features.to_csv('perform_features.csv')

"""
In this section, we get a dataframe that contains the ad performance of certain vehicles with some vehicle attributes from 'vehicles', 
through a merge on the two dataframes. Note that 'new_pricing' is the dataset we used in the pricing model, and it was obtained from 'vehicles' 
after some cleaning. And 'perform_featuers' was obtained earlier after doing feature engineering on the performance dataset.
"""
# Loading 'vehicles' dataset which was cleaned and is now 'new_pricing'
pricing = pd.read_csv('new_pricing.csv', index_col=0)

# Lading the performance dataset
perform = pd.read_csv('perform_features.csv', index_col=0)

# Renaming columns to merge on them
pricing.rename(columns={'vehicle_id':'veh_id', 'dealerID':'dealer_id'}, inplace=True)

# Merging on vehicle_id and dealer_id
merged1 = pd.merge(perform, pricing, how='left', on=['veh_id', 'dealer_id'])

# Dropping duplicate columns
merged2 = merged1.drop(['year_y', 'make_y', 'model_y', 'trim_y', 'vin_y', 'used_y', 'certified_y', 'sold_y', 
'created_at_y', 'deleted_at_y', 'number_days_posted_y'], axis=1)

# Replacing NA in certain columns with 'unknown'
merged2['body'] = merged2['body'].replace(np.nan, 'unknown')
merged2['exterior_color'] = merged2['exterior_color'].replace(np.nan, 'unknown')
merged2['interior_color'] = merged2['interior_color'].replace(np.nan, 'unknown')
merged2['engine'] = merged2['engine'].replace(np.nan, 'unknown')
merged2['trim_x'] = merged2['trim_x'].replace(np.nan, 'unknown')

# Renaming columns back to their original names
all_features_df = merged2.rename({'vin_x':'vin', 'make_x':'make', 'model_x':'model', 'trim_x':'trim',
       'used_x':'used', 'certified_x':'certified', 'year_x':'year', 'created_at_x':'created_at', 'deleted_at_x':'deleted_at',
       'sold_x':'sold', 'number_days_posted_x':'number_days_posted'}, axis='columns')

# Write it to a csv in the future...
#all_features_df.to_csv('all_features_df.csv')

# Read in data, create column for filename of corresponding image
features_df = pd.read_csv('all_features_df.csv')
id_array = features_df.veh_id.tolist()
len(features_df)

def filename(row):
    return('img_' + str(row[2]) + '_.jpg')

features_df['filename'] = features_df.apply(filename, axis=1)

# Fill missing values/data reshaping
# Number of days posted for those that do not have deleted_at date
def fill_date(row):
    today = date.today()
    if pd.isnull(row[20]):
        temp = row[19].split('-')
        d1 = date(int(temp[0]), int(temp[1]), int(temp[2]))
        d0 = date(today.year, today.month, today.day)
        delta = d0 - d1
        return(delta.days)
    else:
        return(row[24])

features_df['number_days_posted'] = features_df.apply(fill_date, axis=1)

# All subsequent text stats need to be 0 for observations with no description
def descprition_stats(row):
    if pd.isnull(row[25]):
        return(0)
    else:
        return(row[indx[i]])

indx = [27, 28, 29, 30, 31]
col_names = ['totalwords', 'uppercase_to_total_words', 'sentiment_score', 'subjectivity_score', 'numeric']
for i in range(len(indx)):
    col = col_names[i]
    features_df[col] = features_df.apply(descprition_stats, axis=1)

# Make used and certified a string
def used(row):
    if row[16] == 1:
        return('used')
    else:
        return('not used')
    
features_df['used'] = features_df.apply(used, axis=1)  

def certified(row):
    if row[17] == 1:
        return('certified')
    else:
        return('not certified')
    
features_df['certified'] = features_df.apply(certified, axis=1)

# Group year - otherwise way too many values for one-hot encoding
def year(row):
    if row[18] == 0 or row[18] == 1:
        return('year unknown')
    if 2020-row[18] >= 10:
        return('greater than 10')
    if 2020-row[18] >= 5 and 2020-row[18] < 10:
        return('between 5 and 10')
    if 2020-row[18] < 5 and 2020-row[18] >= 2:
        return('between 2 and 5')
    if 2020-row[18] < 2:
        return('less than 2')

features_df['year'] = features_df.apply(year, axis=1)

# Mutate response variable - drop impressions that are 0
features_df = features_df[features_df['impressions']!=0].reset_index(drop=True)
def ratio(row):
    ratio = (row[9]/row[6])*100
    return(ratio)

features_df['ratio'] = features_df.apply(ratio, axis=1)

# Train and test sets - 75% test set, 25% train set
stop = int(0.75*len(features_df))
id_array = np.array(features_df.veh_id)
indices = np.random.permutation(id_array.shape[0])
training_idx, test_idx = indices[:stop], indices[stop:]
print(len(test_idx) + len(training_idx))

train_df = features_df.loc[training_idx]
test_df = features_df.loc[test_idx]

# Validation set - 30% of train
j = train_df.index.tolist()
chosen_idx = np.random.choice(j, replace=False, size=int(len(train_df)*.30))
val_df = train_df.loc[chosen_idx]
train_df = train_df.drop(chosen_idx)

print(len(train_df), len(val_df), len(test_df))
print(len(train_df) + len(val_df) + len(test_df))

# Imputing needs to be done only after train/test/val split if missing values still exist
def impute_mileage(x):
    return(x.mileage.fillna(x.mileage.median()))

train_df['mileage'], test_df['mileage'], val_df['mileage'] = impute_mileage(train_df), impute_mileage(test_df), impute_mileage(val_df)

def impute_price(x):
    return(x.price.fillna(x.price.median()))

train_df['price'], test_df['price'], val_df['price'] = impute_price(train_df), impute_price(test_df), impute_price(val_df)

# This will be used in standardizing images for respective sets
def numpy_image(row):
    path = '/home/oroberts/DealersUtd/jpg_images/' + row[45] #your working directory here
    img = np.array(Image.open(path).resize((200,200)))
    return(img)

#np_train = train_df.progress_apply(numpy_image, axis = 1).tolist()
np_test = test_df.progress_apply(numpy_image, axis = 1).tolist()
#np_val = val_df.progress_apply(numpy_image, axis = 1).tolist()

# Full set of categorical levels for one hot encoding
days = list(set(features_df.day_created_at))
used = list(set(features_df.used))
cert = list(set(features_df.certified))
year = list(set(features_df.year))

"""
Mixed input neural network
"""

"""
Most will be commented out except test here for proof of concept. The trained model is saved and will be loaded for predictions, 
but for tuning model in the future, this can be un-commented 
"""

# Standardize images (subtract mean divide by standard deviation)
# Add noise to training images to reduce overfitting

"""
train_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                              rotation_range=45, width_shift_range=.15, height_shift_range=.15)
train_gen.fit(np_train)

val_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
val_gen.fit(np_val)
"""
test_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
test_gen.fit(np_test)

# Flow images from data frame
"""
train_generator = train_gen.flow_from_dataframe(train_df, directory = '/home/oroberts/DealersUtd/jpg_images', 
                                             x_col = 'filename', y_col = 'ratio', class_mode = 'raw', 
                                             shuffle = False, batch_size=len(train_df), target_size=(200,200))

valid_generator = val_gen.flow_from_dataframe(val_df, directory = '/home/oroberts/DealersUtd/jpg_images', 
                                             x_col = 'filename', y_col = 'ratio', class_mode = 'raw', 
                                             shuffle = False, batch_size=len(val_df), target_size=(200,200))
"""

test_generator = test_gen.flow_from_dataframe(test_df, directory = '/home/oroberts/DealersUtd/jpg_images',
                                              x_col = 'filename', y_col = 'ratio', class_mode = 'raw',
                                              shuffle = False, batch_size=len(test_df), target_size=(200,200))

# Append image data to array for model
img_train_x = []
img_train_y = []
img_val_x = []
img_val_y = []
img_test_x = []
img_test_y = []

"""
x, y = train_generator.next()
img_train_x.append(x)
img_train_y.append(y)

x, y = valid_generator.next()
img_val_x.append(x)
img_val_y.append(y)
"""

x, y = test_generator.next()
img_test_x.append(x)
img_test_y.append(y)

# Convert to numpy for tensorflow compatibility
"""
img_train_x = np.array(img_train_x[0])
img_train_y = np.array(img_train_y[0])
img_val_x = np.array(img_val_x[0])
img_val_y = np.array(img_val_y[0])
"""
img_test_x = np.array(img_test_x[0])
img_test_y = np.array(img_test_y[0])

#Pre-process continuous and categorical features
def process_structured_data(df, train, test, val):
    continuous_columns = ['number_days_posted', 'totalwords', 'uppercase_to_total_words', 'sentiment_score', 
                          'subjectivity_score', 'numeric', 'mileage', 'price'] 
    categorical_columns = ['day_created_at', 'used', 'certified', 'year']
    t = [('cat', OneHotEncoder([days, used, cert, year]), categorical_columns), ('num', MinMaxScaler(), continuous_columns)]
    col_transform = ColumnTransformer(transformers=t)
    train_x = col_transform.fit_transform(train)
    test_x = col_transform.fit_transform(test)
    val_x = col_transform.fit_transform(val)
    train_y = np.array(train_df.ratio.tolist())
    val_y = np.array(val_df.ratio.tolist())
    test_y = np.array(test_df.ratio.tolist())
    return(train_x, test_x, val_x, train_y, val_y, test_y)

train_x, test_x, val_x, train_y, val_y, test_y = process_structured_data(features_df, train_df, test_df, val_df)

"""
# MLP structure
def create_mlp(dim, regularizer=None):
    model = Sequential()
    model.add(Dense(20, input_dim=dim, activation="relu", kernel_regularizer=regularizer))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="relu", kernel_regularizer=regularizer))
    model.add(Dropout(0.1))
    model.add(Dense(5, activation="relu", kernel_regularizer=regularizer))
    return model

# CNN structure
def create_cnn(width, height, depth):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding = 'same', activation='relu', input_shape=(height, width, depth)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(64, kernel_size=(5, 5), padding ='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GaussianNoise(0.1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(5, activation='relu'))
    return model 

# Create the MLP and CNN models
mlp = create_mlp(train_x.shape[1])
cnn = create_cnn(200, 200, 3)

# Input to the final set of layers as the output of both the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# Fully-connected layer head has two dense layers
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="relu")(x)

# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value, engagement rate
model1 = Model(inputs=[mlp.input, cnn.input], outputs=x)

# Compile the model 
opt = Adam(lr=1e-3, decay=1e-3/200)
model1.compile(loss='mean_absolute_error', optimizer=opt)

# Implement early stoppping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

# Train the model
model1.fit([train_x, img_train_x], img_train_y, validation_data=([val_x, img_val_x], img_val_y), 
           epochs=20, batch_size=64, callbacks=[es])

# Save model
model1.save('/home/oroberts/DealersUtd') # your working directory here

"""
# Load model
load_model = tensorflow.keras.models.load_model('/home/oroberts/DealersUtd') # your working directory here

"""
# Plot the training/validation history
plt.plot(model1.history.history['loss'])
plt.plot(model1.history.history['val_loss'])
plt.title('Train vs. Validation by Epochs')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('train_val.png')
"""

# Predicting on the test set
# Change to 'model1' if want to predict after training using this code
print('Mixed input NN MAE:', mean_absolute_error(test_y, load_model.predict([test_x, img_test_x])))

"""
# View train validation over epochs
img = mpimg.imread('train_val.png')
imgplot = plt.imshow(img)
plt.show()
"""

"""
Random forest regression
""" 
# Subset data
train_y = train_df[['ratio']]
test_y = test_df[['ratio']]

train_df = train_df[['number_days_posted', 'totalwords', 'uppercase_to_total_words', 'sentiment_score', 
          'subjectivity_score', 'numeric', 'mileage', 'price', 'day_created_at', 'used', 
          'certified', 'year']]

test_df = test_df[['number_days_posted', 'totalwords', 'uppercase_to_total_words', 'sentiment_score', 
          'subjectivity_score', 'numeric', 'mileage', 'price', 'day_created_at', 'used', 
          'certified', 'year']]

# Pre-processing categorical features
categorical_columns = ['day_created_at', 'used', 'certified', 'year']
t = [('cat', OneHotEncoder(categories = [days, used, cert, year]), categorical_columns)]
col_transform = ColumnTransformer(transformers=t, remainder='passthrough')
train_x = col_transform.fit_transform(train_df)
test_x = col_transform.fit_transform(test_df)

# Fitting
clf = tree.DecisionTreeRegressor()
clf = clf.fit(train_x, train_y)

# Predicting on test set
print('Random forest MAE:', mean_absolute_error(test_y, clf.predict(test_x)))

# Feature importance
name_list = []
imp_list = []
for name, importance in zip(train_df.columns, clf.feature_importances_):
    name_list.append(name)
    imp_list.append(importance)

imp_df = pd.DataFrame([name_list, imp_list]).T
imp_df.columns = ['name', 'importance']
imp_df = imp_df.sort_values('importance', ascending = True).reset_index(drop=True)

features = imp_df.columns
importances = imp_df.importance
indices = np.argsort(importances)

"""
plt.title('Feature Importances')
plt.barh(imp_df.name, imp_df.importance, color='b', align='center')
plt.yticks(range(len(imp_df.importance)))
plt.xlabel('Relative Importance')
plt.show()
"""