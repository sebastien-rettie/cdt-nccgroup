# Graph Extraction
Here are all the scripts I used to recover CFGs and CGs from binaries and construct feature matrices.  Binaries go in, all the numpy objects associated with a CFG and CG come out in npz format.

- `angr_graph_extract.py`: this is doing the graph recovery. Do `python angr_graph_extract.py -h` to see arguments.
	- can recover CFG with set opcodes or recording all opcodes for processing later.
	- can recover CG with set API call categories
	- the code is probably hard to follow because of all the error handling and conditionals, message me if you want me to try and explain something.
- `allAPI_categories_techheaders_reduced.yaml`: the categories I ended up using, they are scraped from Microsoft docs. 
- api_categorisation/: this directory contains all the scripts I used to scrape the documentation and build the API call categories yaml file.
- `angr_env.yaml`: conda environment required for graph revovery.

## Workflow:
```
api_categrisation/ scripts --> allAPI_categories_techheaders_reduced.yaml \                             --> CFGs
                                                                           \                           /
                                                                            --> angr_graph_extract.py -
                                                                           /                           \
                                                                 binaries /                             --> CGs
         
```
