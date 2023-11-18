#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#%%
cols="""duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

columns=[]
for c in cols.split(','):
    if(c.strip()):
       columns.append(c.strip())

columns.append('target')
#print(columns)
print(len(columns))
# %%
data = pd.read_csv("kddcup.data.gz", names=columns)
data.head()
# %%
df = data[(data["target"] == "normal.") | (data["target"] == "teardrop.")]
len(df)
# %%
df[df["target"] == "teardrop."].head()
len(df)
# %%
df.head()
# %%
unique_counts = df.nunique()
single_unique_columns = unique_counts[unique_counts == 1].index
df_subset = df.drop(columns=single_unique_columns)
df_subset.head()
# %%
df_subset.to_csv("main.csv", index=False)
# %%
df_subset = pd.read_csv("main.csv")
df_subset.head()
# %%
normal_samples = df_subset[df_subset["target"] == "normal."].sample(n=1000, random_state=42)
teardrop_samples = df_subset[df_subset["target"] == "teardrop."]

df_balanced = pd.concat([normal_samples, teardrop_samples], ignore_index=True)
df_balanced.head()

# %%
len(df_balanced)
df_balanced.to_csv("balanced.csv", index=False)
# %%
len(df_balanced)
# %%
df_balanced = pd.read_csv("balanced.csv")
df_balanced.describe()
# %%
unique_counts = df_balanced.nunique()
single_unique_columns = unique_counts[unique_counts == 1].index
df_balanced.drop(columns=single_unique_columns, inplace=True)
# %%
df_balanced.head()
# %%
df_balanced["service"].unique()
# %%
df_balanced["target"] = df_balanced["target"].replace({"normal.": 0, "teardrop.": 1})
df_balanced["protocol_type"] = df_balanced["protocol_type"].replace({"tcp": 0, "udp": 1, "icmp": 2})
df_balanced["service"] = df_balanced["service"].replace({"smtp": 0, "http": 1, "ftp_data": 2, "private": 3, "http": 4, "domain_u": 5, "finger": 6, "auth": 7, "urp_i": 8, "ecr_i": 9, "ftp": 10, "eco_i": 11, "IRC": 12, "other": 13})
df_balanced["flag"] = df_balanced["flag"].replace({"SF": 0, "REJ": 1, "RSTO": 2, "S0": 3, "RSTR": 4})
# %%
scaler = preprocessing.MinMaxScaler()
d = scaler.fit_transform(df_balanced)
scaled_df = pd.DataFrame(d, columns=df_balanced.columns)
scaled_df.head()
# %%
scaled_df.to_csv("scaled.csv", index=False)

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_df.drop("target", axis=1), scaled_df["target"], test_size=0.05, random_state=42)
# %%
def func_gaus(w, x, g=0.1):
    return np.exp(-sum([(wi - xi)**2 for wi, xi in zip(w, x)]) / (g ** 2))

# %%
delta = 1
y_summation = {0.0 : 0, 1.0 : 0}
#calculating y for the pattern layer
for test_row, y_expected in zip(X_test.values, y_test.values):
    #print(test_row)
    y_pattern = [func_gaus(train_row, test_row) for train_row in X_train.values]
    print("The values of y pattern layer\n", y_pattern)
    #calculating the values of summation layer
    for i in range(len(y_pattern)):
        y_summation[y_train.values[i]] += y_pattern[i] 
    print("The values of y summation layer\n",y_summation)
    #defining the output
    y_output = max(y_summation, key=lambda k: y_summation[k])
    print("The predicted class is", y_output)
    print("The expected class is", y_expected)
    #cleaning the sum values
    y_summation = {key: 0 for key in y_summation}
# %%
