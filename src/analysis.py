import glob 
import pandas as pd
import os
from tqdm import tqdm

all_data = ''
for d in ['mmrec-baby-dataset']:
    log_files = glob.glob(f'./log/*{d}*.log')
    data_test = {}
    data_val = {}
    for log in tqdm(log_files, total=len(log_files), desc="Processing"):
        if d not in log:
            continue
        
        with open(log, 'r') as f:
            data = f.read().strip().split('\n')
        
        for i, l in enumerate(data):
            if 'INFO test result: ' in l:
                try:
                    # Get the next line after 'INFO test result: '
                    l = data[i + 1]
                    l = l.replace('recall', '').replace('ndcg', '').replace('precision', '').replace('map', '').replace(',', '').split('@')[1:]
                    l = [float(i.split(':')[-1]) for i in l]
                except Exception as e:
                    print(f"Error processing log {log} at line {i}: {e}")
                    continue

                if len(l) == 16:
                    # Keep the log entry with the highest sum value
                    if data_test.get(log) == None or (data_test.get(log) != None and sum(data_test[log]) < sum(l)):
                        data_test[log] = l
            
        if data_test.get(log) == None:
            print('NULL', log)
            # Uncomment this if you really want to delete the logs, otherwise leave it out.
            # os.system(f'rm {log}')
            continue
        
    index = []
    for i in ['recall', 'ndcg', 'precision', 'map']:
        for k in [5, 10, 20, 50]:
            index.append(f'{i}@{k}')
            
    data_test = pd.DataFrame(data_test, index=index).T
    data_test = data_test[['recall@5', 'precision@5', 'map@5', 'ndcg@5']]
    data_test["sum"] = data_test.apply(lambda x: x.sum(), axis=1)
    data_test = data_test.sort_values('sum', ascending=False)
    data_test.drop(columns=['sum'], inplace=True)
    
    data_test.to_csv('analysis.txt', sep='\t')
    with open('analysis.txt', 'r') as f:
        all_data += f.read() + '\n\n'

with open('analysis.txt', 'w') as f:
    f.write(all_data)
