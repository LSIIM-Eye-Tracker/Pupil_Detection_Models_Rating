import os
import pandas as pd
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

    # count values between margins
    
    # draw histogram
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(df['center_error_x'], bins=50)
    axs.set_title('Center Error X')
    #fig.show()
    fig.savefig(f"results/{csv_results[analyse_index].split('.')[0]}_center_error_x.png")
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(df['center_error_y'], bins=50)
    axs.set_title('Center Error Y')
    #fig.show()
    fig.savefig(f"results/{csv_results[analyse_index].split('.')[0]}_center_error_y.png")
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(df['radius_error'], bins=50)
    axs.set_title('Radius Error')
    #fig.show()
    fig.savefig(f"results/{csv_results[analyse_index].split('.')[0]}_radius_error.png")
    
    # count results between margins
    df['center_error_x'] = df['center_error_x'].astype(float)
    df['center_error_y'] = df['center_error_y'].astype(float)
    df['radius_error'] = df['radius_error'].astype(float)
    df_margins['e_m_x'].astype(float)
    df_margins['e_m_y'].astype(float)
    df_margins['e_m_r'].astype(float)
    center_error_x = df[(df['center_error_x'] <= df_margins['e_m_x'][0])]["center_error_x"].count()
    print(f"Center Error X under error margin of {df_margins['e_m_x'][0]}: {center_error_x}")
    center_error_y = df[(df['center_error_y'] <= df_margins['e_m_y'][0])]["center_error_x"].count()
    print(f"Center Error Y under error margin of {df_margins['e_m_y'][0]}: {center_error_y}")
    radius_error = df[(df['radius_error'] <= df_margins['e_m_r'][0])]["radius_error"].count()
    print(f"Radius Error under error margin of {df_margins['e_m_r'][0]}: {radius_error}")
    print(f'Total number of results: {df["center_error_x"].count()}')
    #print(df_margins)

    #print(df.describe())