import tensorflow as tf
import os
import pandas as pd

if __name__ == '__main__':
    zip_file = tf.keras.utils.get_file(
        fname="cora.tgz",
        origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(zip_file), "cora")
    print(data_dir)
    
    citations = pd.read_csv(
        os.path.join(data_dir, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )
    
    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"),
        sep="\t",
        header=None,
        names=["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"],
    )
    
    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}
    
    papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
    citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
    citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
    papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

    print(citations)

    print(papers)
