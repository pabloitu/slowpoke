# slowpoke


## Set-up Python environemnt

Using `conda` or `mamba`
```shell
mamba env create -f environment.yml
```

or using `pip`

```shell
python3 -m venv venv
source venv/bin/activate
python install -e .
```


## Building up SLAB2.0

TL;DR

This must be built from a different environment to avoid conflicting libraries/python versions

First, comment the package `sklearn` from `slab2/slab2setup/slab2env.yml` since it is deprecated

Then, to install the env
```shell
conda deactivate
git clone https://github.com/usgs/slab2
cd slab2/slab2setup
bash slab2env.sh
```
To create a slab model, then
```shell
cd slab2/slab2code
python slab2.py -p library/parameterfiles/${SUBZONE_NAME}.par
```

