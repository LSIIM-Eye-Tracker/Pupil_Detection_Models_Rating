import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
if __name__ == "__main__":
    results = os.listdir("results")
    csv_results = []
    for result in results:
        if(result.endswith(".csv")):
            csv_results.append(result)
    for i in range(len(csv_results)):
        print(f"{i} - {csv_results[i]}")
    analyse_index = int(input("Qual resultado deseja analisar? "))
    df = pd.read_csv(f"results/{csv_results[analyse_index]}")
    df_margins = pd.read_csv('results/error_margins/error_margins.csv', index_col=0)
    df_margins = df_margins[df_margins['dataset'] == csv_results[analyse_index].split('.')[0]]

    
    '''# draw histogram
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(df['center_error'], bins=50)
    axs.set_title('Center Error')
    #fig.show()
    
    fig.savefig(f"results/{csv_results[analyse_index].split('.')[0]}_center_error_y.png")
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(df['radius_error'], bins=50)
    axs.set_title('Radius Error')
    #fig.show()
    fig.savefig(f"results/{csv_results[analyse_index].split('.')[0]}_radius_error.png")
    
    # count results between margins
    df['center_error'] = df['center_error'].astype(float)
    df['radius_error'] = df['radius_error'].astype(float)
    df_margins['e_m_center'].astype(float)
    df_margins['e_m_r'].astype(float)
    total = int(df["center_error"].count())
    center_error = df[(df['center_error'] <= df_margins['e_m_center'][0])]["center_error"].count()
    print(f"Center Error under error margin of {df_margins['e_m_center'][0]}: {center_error} ({100*center_error/total:.2f}%)")
    radius_error = df[(df['radius_error'] <= df_margins['e_m_r'][0])]["radius_error"].count()
    print(f"Radius Error under error margin of {df_margins['e_m_r'][0]}: {radius_error} ({100*radius_error/total:.2f}%)")
    print(f'Total number of results: {total}')'''

    columns = ['TP','FP','TN','FN','TPR','FPR','TNR','FNR','ACC','sensitivity','specificity','weighted_acc']
    for column in tqdm(columns):
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(df[column], bins=50)
        axs.set_title(column)
        fig.savefig(f"results/{csv_results[analyse_index].split('.')[0]}_{column}.png")

    desc = df.describe()
    desc.to_csv(f"results/descriptions/{csv_results[analyse_index].split('.')[0]}_desc.csv")

    #print(df_margins)

    #print(df.describe())