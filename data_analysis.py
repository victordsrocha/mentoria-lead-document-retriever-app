import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    df = pd.read_csv(f'data/no_english/DAL_Corpus_no_english.csv', sep=';')

    u, c = np.unique(df['curso'].values.astype(str), return_counts=True)
    print(f'Quantidade de amostras: {df.shape[0]}')
    print(f'Quantidade de classes: {len(u)}')

    plot = sns.barplot(x=list(range(1, len(u) + 1)), y=c)
    fig = plot.get_figure()
    fig.savefig('./figuras/courses_barplot.svg')

    for i in zip(u, c):
        print(i)


if __name__ == '__main__':
    main()
