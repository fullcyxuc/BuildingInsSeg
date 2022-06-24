# Buliding Instance Segmentation for Software Test

## Install 3DBuildingInstanceSeg

(1) Clone from the repository.
```
git clone https://github.com/fullcyxuc/BuildingInsSeg.git
cd BuildingInsSeg
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost  # or boost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) We also use other cuda and cpp extension([pointgroup_ops](https://github.com/dvlab-research/PointGroup/tree/master/lib/pointgroup_ops),[pcdet_ops](https://github.com/yifanzhang713/IA-SSD/tree/main/pcdet/ops)), to compile them:
```
cd lib/**  # (** refers to a specific extension)
python setup.py develop
```

## Usage

(1) data and model
* test examples have been put into the dir `exp`, test files end up with `.txt`
* in `val_gt` dir, there are the instance GT infos for the corresponding test files
* `*.pth` is the model file

(2) instantiate the class `InsSegTest` and run the function `test()` with a given test file path in `test.py`:
```
t = InsSegTest()
result = t.test(data_path='exp/6_wanxia_0_66.txt')
```

(3) more information about the return have been described in function `test()`
