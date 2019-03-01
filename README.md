# distil-vector-to-cols

Given the index of a column containing vector data of length N, creates N new columns, each containing the data from the corresponding vector element.  Labels for each of the new columns can be optionally specified, other wise the name of the original column will be used with a number affixed (someVectorCol -> someVectorCol_0, someVectorCol_1).  If the vectors stored in the source column of are of differing lengths, the operation will fail.

Deployment:

```shell
pip install -e git+ssh://git@github.com/uncharteds-distil/distil-vector-to-cols.git#egg=DistilVectorToCols --process-dependency-links
```

Development:

```shell
pip install -r requirements.txt
```
