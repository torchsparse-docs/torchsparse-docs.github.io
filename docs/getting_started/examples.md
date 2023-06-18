  # Examples for TorchSparse v2.1

  We also made it easier to integrate our library to other 3D deep learning algorithm frameworks or implementing new models from scratch in TorchSparse v2.1. Here we showcase two examples (`mmdetection3d` integration and implementing models from scratch in `spvnas`):

  ## Mmdetection3d integration

  ### Overview

  We present a end-to-end example of TorchSparse v2.1 integration into mmdetection3d in the [BEVFusion](https://github.com/mit-han-lab/bevfusion/tree/dev/torchsparsepp_backend) repository. BEVFusion is a multi-modal 3D perception model that takes in both surrounding camera images and LiDAR scans and fuses these information in the shared bird's-eye-view space. The SparseConv-based LiDAR feature extractor is the key to BEVFusion's performance, which is well-supported by TorchSparse v2.1.

  One may want to pay attention to the following files when investigating our integration, since it might also be helpful for potential TorchSparse integration into other frameworks such as Det3D and OpenPCDet.

  - `mmdet3d/models/backbones/sparse_encoder.py`: SparseConv model implementation;

  - `mmdet3d/ops/spconv/__init__.py`: Op registry;

  - `mmdet3d/ops/sparse_block.py`: SparseConv blocks (e.g. plain residual, bottleneck, etc.);

  - `tools/convert_checkpoints_to_torchsparse.py`: checkpoint converter.


  ### Deep dive

  [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) is a widely-used codebase for 3D perception tasks. It seamlessly support different benchmarks and state-of-the-art models. All the models and datasets in mmdetection3d can be easily built with a configuration dictionary. 
  
  #### Model registries

  In TorchSparse v2.1, we present a simple way to integrate our APIs into mmdetection3d. One can start from `mmdet3d/ops/spconv`, which provides native support for the SpConv library. After deleting all the files in this directory, one can create a new `__init__.py` and add the following content:


  ```python
  import itertools

  from mmcv.cnn.bricks.registry import CONV_LAYERS, NORM_LAYERS
  from torch.nn.parameter import Parameter

  def register_torchsparse():
      """This func registers torchsparse ops."""
      from torchsparse.nn import Conv3d, BatchNorm

      CONV_LAYERS._register_module(Conv3d, "TorchSparseConv3d", force=True)
      NORM_LAYERS._register_module(BatchNorm, "TorchSparseBatchNorm", force=True)

  register_torchsparse()
  ```

#### Building blocks

  That's it! When you are defining any `nn.Module` in PyTorch, you can construct a TorchSparse v2.1 convolution layer using:


  ```python
  from mmcv.cnn import build_conv_layer
  conv_cfg = dict(type="TorchSparseConv3d")
  layer = build_conv_layer(
      conv_cfg,
      in_channels,
      out_channels,
      kernel_size,
      stride=stride,
      padding=padding,
      bias=False,
  )
  ```

  Note that `build_conv_layer` is implemented by [mmcv](https://github.com/open-mmlab/mmcv). We do not even need to modify the model registry by ourselves. 

  Another interesting fact about TorchSparse v2.1 is that one can almost fully reuse existing `mmcv` implementation of 2D CNN building blocks. Here is an example for the plain residual block:

  ```python
  class SparseBasicBlock(BasicBlock):

      expansion = 1

      def __init__(
          self,
          inplanes,
          planes,
          stride=1,
          downsample=None,
          conv_cfg=None,
          norm_cfg=None,
          act_cfg=None,
      ):
          BasicBlock.__init__(
              self,
              inplanes,
              planes,
              stride=stride,
              downsample=downsample,
              conv_cfg=conv_cfg,
              norm_cfg=norm_cfg,
          )
          if act_cfg is not None:
              if act_cfg == "swish":
                  self.relu = spnn.SiLU(inplace=True)
              else:
                  self.relu = spnn.ReLU(inplace=True)
  ```

  Notice that one only needs to modify the `self.relu` function in the original plain residual block implementation. This is because `self.relu` has to take in `torchsparse.SparseTensor`s. We do not even need to touch the `forward` function of this module thanks to the fact that the forward pass in TorchSparse is identical to that in PyTorch.

#### Migrating from SpConv

  Given that mmdetection3d provides existing implementations for the SpConv backend as well, converting to TorchSparse will not be too difficult. We provide an easy-to-use checkpoint converter [here](https://github.com/mit-han-lab/bevfusion/tree/dev/torchsparsepp_backend/tools/convert_checkpoints_to_torchsparse.py) to directly convert SpConv-pretrained models to TorchSparse models. The inference results will be exactly the same after conversion. For model level transformations, all you need to do is to modify the model registry from

  ```python
  conv_cfg = dict(type="SparseConv3d")
  ```

  which registers to the SpConv backend to 

  ```python
  conv_cfg = dict(type="TorchSparseConv3d")
  ```

  which corresponds to our backend. Notice that you do not need to specify fields like `indice_key` in SpConv. These fields are automatically handled by TorchSparse v2.1.

  ## Building models from scratch

  ### Overview

  Alternatively, it is not difficult to build new models from scratch without model registries. We showcase such application in [SPVNAS](https://github.com/mit-han-lab/spvnas/tree/dev/torchsparsepp_backend). SPVNAS is an early work attempting to automatically design efficient architectures for 3D semantic segmentation. If is composed of Sparse Point-Voxel Convolution (SPVConv) layers, which is natively supported by TorchSparse. Here are some example files that one might be interested in when building models from scratch:

  - `core/models/semantic_kitti/minkunet.py`: defining the most basic [MinkowskiUNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) (CVPR 2019) architecture;

  - `core/models/utils.py`: Updated voxelization / devoxelization implementation in TorchSparse v2.1;

  - `core/models/semantic_kitti/spvcnn.py`, `core/models/semantic_kitti/spvnas.py`: Implementation for SPVCNN and SPVNAS.

  ### Deep dive

  Implementing new sparse models using TorchSparse v2.1 is incredibly easy. 
  
  #### Pure SparseConv models
  
  The high-level guideline is that you **do not need to change anything** compared with previous TorchSparse implementation if you are using pure SparseConv-based architectures (like MinkUNet or the SECOND / CenterPoint encoders). The following global flags will help you achieve better speedup or have control over the model's behavior in downsampling:

  ```python
  import torchsparse.nn.functional as F

  # Setting conv mode
  # Available choices are 0, 1, 2, which are optimized for different workloads and achieve close performance to the autotuned configurations
  # 2 is recommended for semantic segmentation

  F.set_conv_mode(2)

  # Select the way how kernel maps are constructed.
  # "hashmap" is compatible with MinkowskiEngine and "hashmap_on_the_fly" is compatible with SpConv. 
  # The results are not too different using these two but "hashmap_on_the_fly" is slightly faster in most cases.

  F.set_kmap_mode("hashmap")

  # Select the compatible mode for downsampling.
  # Choosing between "spconv" and "minkowski". 
  # The same behavior for kernel_size = stride cases, 
  # SpConv gives larger output size when kernel_size > stride.

  F.set_downsample_mode("spconv")
  ```

  Note: if these fields are not set, they default to 

  ```python
  conv_mode = 1
  kmap_mode = "hashmap_on_the_fly"
  downsample_mode = "spconv"
  ```

  For some applications like 3D reconstruction, it might be necessary to set downsample mode to "minkowski" to achieve the best performance.

#### SPVConv modifications

  For modules like SPVConv where voxelization and devoxelization operators are involved, the implementation differs from TorchSparse <= 2.0. We refer the users to the new implementation of `core/models/utils.py` versus the old one in the master branch. In short, the fact that:

  - We change the coordinate representation, moving batch index to the first dimension and always shrink the scale of coordinates after downsampling;
  - We deprecate several inefficient hashmap APIs

  requires us to reimplement these operators. But the modifications are straightforward and have been done on our side.


