import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def read_data_cites():
    citations = pd.read_csv('cora/cora.cites', sep='\t',
                            names=['target', 'source'], header=None)
    papers = pd.read_csv('cora/cora.content', sep='\t', header=None,
                         names=['paper_id']+['feature_'+str(i) for i in range(1433)]+['subject'])

    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(
        sorted(papers["paper_id"].unique()))}
    

    papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
    citations["source"] = citations["source"].apply(
        lambda name: paper_idx[name])
    citations["target"] = citations["target"].apply(
        lambda name: paper_idx[name])
    papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])
    return citations, papers


if __name__ == '__main__':
    citations, papers = read_data_cites()
    print(citations)
    print(papers)

    plt.figure(figsize=(10, 10))
    colors = papers["subject"].tolist()
    cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
    subjects = list(papers[papers["paper_id"].isin(
        list(cora_graph.nodes))]["subject"])
    nx.draw_spring(cora_graph, node_size=15, node_color=subjects)
    plt.show()
