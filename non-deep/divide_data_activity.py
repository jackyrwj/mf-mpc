import pandas as pd

df = pd.read_csv('MF-MPC-dataset-ML100K/copy1.train',sep = ' ',names = ['user','item','rate'])
df_filter = {}
df_filter[0] = df.groupby('user').filter(lambda x:(len(x) <= 20))
df_filter[1] = df.groupby('user').filter(lambda x:(len(x) > 20 and len(x) <= 50))
df_filter[2] = df.groupby('user').filter(lambda x:(len(x) > 50))

for i in range(0, 3):
    save_data = df_filter[i]
    file_name =  str(i) + '.train'  
    save_data.to_csv(file_name, index=False, header = 0, sep = ' ')  
