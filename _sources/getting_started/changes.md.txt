# Changes in TorchSparse v2.1

In TorchSparse v2.1, we introduced and optimized the Implicit-GEMM dataflow to improve the computing efficiency of Sparse Convolution. 

## TF32

We support TF32 precision in TorchSparse v2.1. Use following code to activate TF32 mode:

```python
import torchsparse.backends
torchsparse.backends.allow_tf32 = True	# True by default on NVIDIA GPUs >= SM80
```

## User API

The APIs for Sparse Convolution in TorchSparse v2.1 is PyTorch-like and can be easily accessed in `torchsparse.nn` :

```python
from torchsparse import nn as spnn
SparseConvLayer = spnn.Conv3d(in_channels, out_channels, kernel_size)
```

The input arguments for `spnn.Conv3d` are as follows:

| Input Argument |               Type                | Mandatory | Default Value |
| :------------: | :-------------------------------: | :-------: | :-----------: |
|  in_channels   |                int                |    Yes    |       /       |
|  out_channels  |                int                |    Yes    |       /       |
|  kernel_size   | int / List[int] / Tuple[int, ...] |    No     |       3       |
|     stride     | int / List[int] / Tuple[int, ...] |    No     |       1       |
|    padding     | int / List[int] / Tuple[int, ...] |    No     |       0       |
|    dilation    |                int                |    No     |       1       |
|      bias      |               bool                |    No     |     False     |
|   transposed   |               bool                |    No     |     False     |
|   generative   |               bool                |    No     |     False     |

* The argument `generative` is newly introduced in TorchSparse v2.1. If both `generative`  and `transposed` are `True` , the conv module will execute generative transposed convolution.  

## Conv_mode & Auto-tune

### Sparse Conv Mode

We provide 3 different **Sparse Convolution Modes** in TorchSparse v2.1 to accommodate different workloads (get better performance). The conv_mode can be get/set through:

``` python
from torchsparse.nn import functional as F

F.set_conv_mode(0)				# Set conv_mode to 0/1/2
conv_mode = F.get_conv_mode()
print(conv_mode)
```

* **conv_mode can significantly impact the performance of Sparse Convolution**.  

* Generally, `conv_mode = 0` (default) is more suitable for workloads with lower sparsity (e.g., detection tasks) and lower compute precision (e.g., FP16), while `conv_mode = 1 or 2` might be a better choice for higher compute precision and sparsity.

* Please refer to the following table below for recommended `conv_mode` settings. However, you are always encouraged to experiment and evaluate the speed differences to find the optimal setting for your specific use case：

  |                  |     FP16      |     TF32      |     FP32      |
  | :--------------: | :-----------: | :-----------: | :-----------: |
  | **Segmentation** | `conv_mode 0` | `conv_mode 2` | `conv_mode 2` |
  |  **Detection**   | `conv_mode 0` | `conv_mode 0` | `conv_mode 2` |


### Sparse Auto-tune (Experimental Feature)

We also provide Sparse Auto-tuner in TorchSparse v2.1 to optimize **Sparse Conv Settings** autonomously.  To enable Auto-optimization, you need to first run the model with a few samples to tune conv kernel configurations:

```python
model = ... # your model to be used
dataflow = ... # an iterator with your data samples, usually in the form of torch.utils.data.DataLoader
save_dir = ... # path to save tuning results, a str
tune_tag = ... # tag for tuning, a str

torchsparse.tune(
    model=model,
    data_loader=dataflow,
    n_samples=100,				 # How many data samples to use in auto-tuning
    collect_fn=lambda data:data, 
    enable_fp16 = True, 
    save_dir = save_dir,
    tune_tag = tune_tag,
    force_retune = True, 		 # Whether to ignore cached tuning results
    tune_with_bwd = False,		 # Whether to tune for backward kernels
)

```

* `collect_fn`: Process data before calling model.forward(). Or rather, run`model(*collect_fn(data))` where data is yielded by `data_loader`. The default case handles `{'input': SparseTensor,...}` for data.



## Sparse Mapping Mode

We provided two **sparse mapping modes** in TorchSparse v2.1: `hashmap` and `hashmap_on_the_fly` (default). The `grid` mode in previous TorchSparse has been deprecated.

TorchSparse v2.1 relies on hash method to map input & output points. The `hashmap_on_the_fly` mode fuses operations (e.g., build hash table, hash query) together, making it slightly faster than `hashmap` mode in most cases.

The **sparse mapping modes** can be easily set through:

```python
from torchsparse.nn import functional as F

F.set_kmap_mode('hashmap')		# Choose 'hashmap' or 'hashmap_on_the_fly'
kmap_mode = F.get_kmap_mode()
print(kmap_mode)
```

## Downsample Mode

TorchSparse v2.1 is compatible with the two different downsampling modes in [SpConv](https://github.com/traveller59/spconv) and [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine):

```python
from torchsparse.nn import functional as F

F.set_downsample_mode('spconv')		# Choose 'spconv' or 'minkowski'
downsample_mode = F.get_downsample_mode()
print(downsample_mode)
```


## Additional Notes

### Coordinate Order - Batch First

* We changed the coordinate order from `[x, y, z, batch]`  to `[batch, x, y, z]` in TorchSparse v2.1.

### Negative coordinates

* TorchSparse v2.1 allows negative coordinates as inputs.

  ``` python
  import torchsparse.tensor as ts_ten
  
  ts_ten.set_allow_negative_coordinates(True)			# False by default	
  neg_flag = ts_ten.get_allow_negative_coordinates()
  print(neg_flag)
  ```

* Note that allowing negative coordinates can add to the sparse mapping overhead. We recommend to leave this flag as `False` if you can make sure that there will be no negative coordinates in your input tensor.

### SparseTensor.to_dense()

* We designed customized CUDA kernels to efficiently convert `SparseTensor` to its dense counterpart. Simply call `SparseTensor.dense()` to do the conversion.

### Range of coordinates

* TorchSparse v2.1 changes the behavior of downsampling layers. The range of coordinates will shrink by `s` times if we apply a convolution layer with stride = `s`. 

Originally, the coordinate operation can be understood as:

```python
coords_new = torch.floor(coords_old / s).int() * s
```

Now it is similar to 

```python
coords_new = torch.floor(coords_old / s).int()
```

We also support padding to manipulate the range of coordinates during downsampling. The definition of padding is fully compatible with SpConv and will not take effect if `downsample_mode = "minkowski"`. Padding determines the lower and upper bound of coordinates.

