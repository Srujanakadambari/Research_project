# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# %%
# Load the dataset
dataset = pd.read_csv("/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv")
X = dataset.iloc[:3000, 0:7].values
y = dataset.iloc[:3000, 7].values

# %%
########## Label Encoding categorical data  ###########
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
encoder= LabelEncoder()
X[:,6] = encoder.fit_transform(X[:, 6])
y = encoder.fit_transform(y)
#y_original = encoder.inverse_transform(y_encoded)
y = to_categorical(y)

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# %%
# Feature Scaling (To scale all variables to similar scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
########### Part2 (DEEP NEURAL NETWORK)/Multiple Layer Perceptron(MLP)#########################
from keras.models import Sequential  ##module to initialize ANN
from keras.layers import Dense #module required to build layers

DNN_model = Sequential()
DNN_model.add(Dense(input_dim = 7, units = 8, kernel_initializer = 'uniform', activation = 'relu'))
DNN_model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
DNN_model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
DNN_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(DNN_model.summary())

history = DNN_model.fit(X_train, y_train, batch_size = 5,  epochs = 200,
                        validation_data=(X_test, y_test), shuffle=True)
print(history.history.keys())

# %%
################### Part3 (Making predictions and evaluating the model)#################
y_pred = DNN_model.predict(X_test) # Predicting the Test set results
y_pred = (y_pred > 0.95)
y_pred_label = encoder.inverse_transform(np.argmax(y_pred, axis=1))
print(y_pred_label)

y_test_label = encoder.inverse_transform(np.argmax(y_test, axis=1))
print(y_test_label)


# %%
#Making confusion matrix that checks accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_label, y_pred_label)


# %%
#### Visualizing Confusion Matrix  ########
cm_fig = pd.DataFrame(cm, columns=np.unique(y_test_label), index=np.unique(y_test_label))
sb.set(font_scale=1.3)
sb.heatmap(cm_fig, cmap="RdBu_r", annot=True, annot_kws={"size":15}, fmt='.2f')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')

# %%
######### Predicting with new data ##########
new_pred1 = DNN_model.predict(sc.transform(np.array([[5, 6.2, 92, 93, 85, 16, 1]])))
new_pred1_original = encoder.inverse_transform([np.argmax(new_pred1)])
print(new_pred1_original)

new_pred2 = DNN_model.predict(sc.transform(np.array([[0, 3.5, 0, 95, 96, 12, 1]])))
new_pred2_original = encoder.inverse_transform([np.argmax(new_pred2)])
print(new_pred2_original)

new_pred3 = DNN_model.predict(sc.transform(np.array([[3.7, 0.2, 85, 65, 78, 7, 0]])))
new_pred3_original = encoder.inverse_transform([np.argmax(new_pred3)])
print(new_pred3_original)

# %%
!pip install shap


# %%
import shap
import numpy as np


# %%
# Create a SHAP explainer using the DeepExplainer
explainer = shap.DeepExplainer(DNN_model, X_train)


# %%
explainer = shap.DeepExplainer(DNN_model, X_train)


# %%
shap_values = explainer.shap_values(X_test)


# %%
shap.summary_plot(shap_values, X_test)


# %%
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define feature names
feature_names = ['S1(Amp)', 'S2(Amp)', 'S1(Volt)', 'S2(Volt)', 'Light(kiloLux)', 'Temp(degC)', 'Weather', 'State']

# Create a SHAP explainer using the DeepExplainer
explainer = shap.DeepExplainer(DNN_model, X_train)

# Generate SHAP values for a sample of data
shap_values = explainer.shap_values(X_test)

# Create a bar plot for each feature
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar')
plt.show()



# %%
!pip install lime


# %%
import lime
import lime.lime_tabular


# %%
# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['Normal', 'Open', 'Line-line'], discretize_continuous=False)


# %%
import lime
import lime.lime_tabular
import numpy as np

# Define a function to predict probabilities using the model
def predict_proba_wrapper(X):
    # Predict probabilities using the model
    y_pred = DNN_model.predict(X)
    # Return probabilities for each class
    return y_pred

# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['Normal', 'Fault1', 'Fault2'], discretize_continuous=False)

# Explain a prediction for a specific instance
instance_index = 0  # Index of the instance you want to explain
explanation = explainer.explain_instance(X_test[instance_index], predict_proba_wrapper, num_features=len(feature_names))


# %%
# Visualize the explanation
explanation.show_in_notebook()

# Create a DataFrame for seaborn
df = pd.DataFrame(data=shap_values_reshaped, columns=feature_names)

# Generate summary plot as a beeswarm plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_reshaped, df, plot_type='dot', feature_names=feature_names)
plt.show()

# Define the index corresponding to the median prediction
median_index = len(y_pred) // 2

# Create a waterfall plot for the median prediction
plt.figure(figsize=(12, 8))
for output_idx in range(shap_values.shape[-1]):
    shap.waterfall_plot(shap_values[median_index, :, output_idx], max_display=99, show=False)
    plt.title("Waterfall Plot for Median Prediction")
    plt.tight_layout()

plt.show()


# Plot the SHAP values for the median prediction
plt.figure(figsize=(12, 8))
plt.barh(feature_names, shap_values_median_reshaped[0])
plt.xlabel('SHAP Value')
plt.ylabel('Feature')
plt.title('SHAP Local Bar Plot for Median Prediction')
plt.show()

# Plot the SHAP values for the first sample
plt.figure(figsize=(10, 6))
for i in range(len(feature_names)):
    shap_value = shap_values_first_sample[i]
    if isinstance(shap_value, np.ndarray):
        shap_value = shap_value[0]  # Take the first value if it's an array
    color = 'r' if shap_value >= 0 else 'b'
    plt.barh(y=feature_names[i], width=shap_value, color=color)
    plt.text(shap_value, i, '{:.2f}'.format(float(shap_value)), ha='left' if shap_value >= 0 else 'right', va='center')
plt.xlabel('SHAP Value')
plt.ylabel('Feature')
plt.title('SHAP Values for the First Sample')
plt.gca().invert_yaxis()  # Reverse y-axis
plt.show()

# Plot the SHAP values for the middle sample
plt.figure(figsize=(10, 6))
for i in range(len(feature_names)):
    shap_value = shap_values_middle_sample[i]
    if isinstance(shap_value, np.ndarray):
        shap_value = shap_value[0]  # Take the first value if it's an array
    color = 'r' if shap_value >= 0 else 'b'
    plt.barh(y=feature_names[i], width=shap_value, color=color)
    plt.text(shap_value, i, '{:.2f}'.format(float(shap_value)), ha='left' if shap_value >= 0 else 'right', va='center')
plt.xlabel('SHAP Value')
plt.ylabel('Feature')
plt.title('SHAP Values for the mean value of Sample')
plt.gca().invert_yaxis()  # Reverse y-axis
plt.show()

# Plot the SHAP values for the last sample
plt.figure(figsize=(10, 6))
for i in range(len(feature_names)):
    shap_value = shap_values_last_sample[i]
    if isinstance(shap_value, np.ndarray):
        shap_value = shap_value[0]  # Take the first value if it's an array
    color = 'r' if shap_value >= 0 else 'b'
    plt.barh(y=feature_names[i], width=shap_value, color=color)
    plt.text(shap_value, i, '{:.2f}'.format(float(shap_value)), ha='left' if shap_value >= 0 else 'right', va='center')
plt.xlabel('SHAP Value')
plt.ylabel('Feature')
plt.title('Local SHAP plot at 100th percentile value')
plt.gca().invert_yaxis()  # Reverse y-axis
plt.show()
