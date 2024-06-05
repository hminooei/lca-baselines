```python
import numpy as np
from pyscf import gto, scf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc.tools.pbc import get_bandpath
from pyscf.pbc.tools.pbc import get_kmesh
from pyscf.pbc.tools.pbc import get_2d_kmesh
from pyscf.pbc.tools.pbc import get_2d_bandpath
from pyscf.pbc.tools.pbc import get_monkhorst_pack
from pyscf.pbc.tools.pbc import get_bandpath_2d
from pyscf.pbc.tools.pbc import get_bandpath_3d
from pyscf.pbc.tools.pbc import get_bandpath_kpts
from pyscf.pbc.tools.pbc import get_bandpath_kpts_2d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_3d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_4d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_5d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_6d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_7d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_8d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_9d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_10d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_11d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_12d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_13d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_14d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_15d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_16d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_17d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_18d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_19d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_20d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_21d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_22d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_23d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_24d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_25d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_26d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_27d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_28d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_29d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_30d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_31d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_32d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_33d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_34d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_35d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_36d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_37d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_38d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_39d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_40d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_41d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_42d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_43d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_44d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_45d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_46d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_47d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_48d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_49d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_50d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_51d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_52d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_53d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_54d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_55d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_56d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_57d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_58d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_59d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_60d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_61d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_62d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_63d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_64d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_65d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_66d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_67d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_68d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_69d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_70d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_71d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_72d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_73d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_74d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_75d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_76d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_77d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_78d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_79d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_80d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_81d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_82d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_83d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_84d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_85d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_86d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_87d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_88d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_89d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_90d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_91d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_92d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_93d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_94d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_95d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_96d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_97d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_98d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_99d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_100d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_101d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_102d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_103d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_104d
from pyscf.pbc.tools.pbc import get_bandpath_kpts_