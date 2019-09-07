# Datasets

The following datasets can be download with the `pycox.datasets` module:

- `flchain`: Assay of Serum Free Light Chain. \[1\] \[[data source](https://github.com/vincentarelbundock/Rdatasets)\]
  
- `gbsg`: Rotterdam & German Breast Cancer Study Group. \[2\] \[[data source](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data)\]

- `kkbox_v1`: WSDM - KKBox's Churn Prediction Challenge 2017 \[1\]. \[[data source](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)\]

- `metabric`: The Molecular Taxonomy of Breast Cancer International Consortium \[2\]. \[[data source](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data)\]

- `nwtco`: Data from the National Wilm's Tumor \[[data source](https://github.com/vincentarelbundock/Rdatasets)\]

- `support`: Study to Understand Prognoses Preferences Outcomes and Risks of Treatment \[2\]. \[[data source](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data)\]


## Example
The follwing code will download the `metabric` dataset and load it in the form of a pandas dataframe

```python
from pycox import datasets
df = datasets.metabric.read_df()
```
## References 

  \[1\] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129):1–30, 2019. \[[paper](http://jmlr.org/papers/v20/18-424.html)\]

  \[2\] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18(1), 2018. \[[paper](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1)\]
