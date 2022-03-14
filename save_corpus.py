import os
import pandas as pd


def main():
    path = "./data"
    df = pd.read_csv(os.path.join(path, "original/DAL_Corpus.csv"), sep=";")

    df.drop("Unnamed: 0", axis=1, inplace=True)

    ind_to_filter = [i for i, c in enumerate(df["curso"]) if "ingles" in c]
    df.drop(labels=ind_to_filter, axis=0, inplace=True)

    df.to_csv(os.path.join(path, "no_english/DAL_Corpus_no_english.csv"), sep=';', index=False)


if __name__ == "__main__":
    main()
