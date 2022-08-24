#!/usr/bin/env python3

"""generate idxn conv kernels dynamically
"""

import os
import sys
import hashlib

def CeilDiv(x, y):
    return -(x // -y)

class KernelInfo:
    def __init__(self, path, s_size, k_num, cta_y_num, cta_x_num, warp_y, warp_x):
        self.path = path
        self.s_size = s_size
        self.k_num = k_num
        self.k_size = self.k_num * self.s_size

        self.cta_y_num = cta_y_num
        self.cta_x_num = cta_x_num

        self.warp_y = warp_y
        self.warp_x = warp_x

        self.cta_y = self.cta_y_num * self.warp_y
        self.cta_x = self.cta_x_num * self.warp_x

        self.kconfig = "_b" + str(self.cta_y) + "x" + str(self.cta_x) + \
                "_w" + str(self.warp_y) + "x" + str(self.warp_x) + \
                "_k" + str(self.k_size) + "_s" + str(self.s_size)

        self.kname = "nvIdxnSm75Int8Conv_imma8816_nhwc" + self.kconfig
        self.fname = "kernels" + "/idxn"  + self.kconfig + ".cu"

        self.WARP_SIZE = 32
        self.MMA_Y = 8
        self.MMA_K = 16
        self.MMA_X = 8
        self.MMA_Y_HALF = self.MMA_Y / 2

        self.INT4_TO_4INT = 4
        self.INT4_TO_16BYTE = 16
        self.INT8_SIZE = 1
        self.PB_PER_TURING_SM = 4

        self.CPI_IMMA8816 = 8.06
        self.CPI_L1_LDG128 = 8
        self.IMMA_LATENCY = 14
        self.DRAM_LATENCY = 220

        self.MAX_REG_NUM_PER_THD = 255
        self.MAX_REG_NUM_PER_CTA = 65536
        self.MAX_SMEM_SIZE_PER_CTA = 49152

        self.cta_num = cta_y_num * cta_x_num
        self.cta_size = self.cta_num * self.WARP_SIZE

        self.dAvn_size = self.warp_y / self.MMA_Y
        self.dBvn_size = self.warp_x / self.MMA_X

    def GetCompGmemRatio(self):
        pb_num_per_cta = self.cta_num if self.cta_num < self.PB_PER_TURING_SM else self.PB_PER_TURING_SM

        cycles_imma = (self.CPI_IMMA8816 * (self.cta_y / self.MMA_Y) * (self.cta_x / self.MMA_X) * (self.s_size / self.MMA_K) / pb_num_per_cta) + self.IMMA_LATENCY

        cycles_ldg = self.CPI_L1_LDG128 * CeilDiv( (self.cta_y + self.cta_x) * self.s_size * self.INT8_SIZE, (self.INT4_TO_16BYTE * self.WARP_SIZE) ) + self.DRAM_LATENCY

        comp_gmem_ratio = cycles_imma / cycles_ldg

        return comp_gmem_ratio

    def GetSMemUsage(self):

        return (self.cta_y + self.cta_size) * self.INT4_TO_4INT

    def GetRegUsage(self):
        ret = 0

        reg_c_v1 = CeilDiv(self.cta_y * self.cta_x, self.cta_size)

        mma_m_num = self.warp_y // self.MMA_Y
        mma_n_num = self.warp_x // self.MMA_X

        mma_m_num_x2 = mma_m_num * 2 # for 16x8x8 shape

        if self.s_size == 16: # c4 type
            reg_a_v1 = mma_m_num_x2 * 1
            reg_b_v1 = mma_n_num    * 1
        elif self.s_size == 32: # c8 type
            reg_a_v1 = mma_m_num_x2 * 2
            reg_b_v1 = mma_n_num    * 2
        elif self.s_size == 64: # c16 type
            reg_a_v1 = mma_m_num_x2 * 4
            reg_b_v1 = mma_n_num    * 4

        reg_a_idx = (mma_m_num_x2 + 1) * self.INT4_TO_4INT
        reg_b_idx = mma_n_num * 2

        reg_c_idx = mma_n_num * 2 + 4

        reg_common_idx = 20

        ret = reg_a_v1 + reg_b_v1 + reg_c_v1 + max(reg_a_idx + reg_b_idx, reg_c_idx) + reg_common_idx

        return ret

    def IsKernelFeasible(self):
        if self.cta_size > 512 or self.cta_size < 64:
            return False

        reg_usage_per_thd = self.GetRegUsage()
        reg_usage_per_cta = reg_usage_per_thd * self.cta_size
        if reg_usage_per_thd > self.MAX_REG_NUM_PER_THD or reg_usage_per_cta > self.MAX_REG_NUM_PER_CTA:
            return False

        smem_usage = self.GetSMemUsage()
        if smem_usage > self.MAX_SMEM_SIZE_PER_CTA:
            return False

        comp_gmem_ratio = self.GetCompGmemRatio()
        if comp_gmem_ratio >= 2 or comp_gmem_ratio <= 0.5:
            return False

        return True


    def GenKernel(self):
        f = open(os.path.join(self.path, self.fname), "w")
        f.write("#define TILE_N_PER_MMA       %d\n" % self.MMA_X)
        f.write("#define TILE_K_PER_MMA       %d\n" % self.MMA_K)
        f.write("#define TILE_M_PER_MMA       %d\n\n" % self.MMA_Y)

        f.write("#define BLK_M_PER_MMA        %d\n" % (self.MMA_Y / 8))
        f.write("#define BLK_N_PER_MMA        %d\n" % (self.MMA_X / 8))

        f.write("#define TILE_N_PER_CTA       %d\n" % self.cta_x)
        f.write("#define TILE_M_PER_CTA       %d\n\n" % self.cta_y)

        f.write("#define TILE_N_PER_WARP      %d\n" % self.warp_x)
        f.write("#define TILE_M_PER_WARP      %d\n\n" % self.warp_y)

        f.write("#define TILE_K_PER_CTA       %d\n" % self.k_size)
        f.write("#define TILE_K_PER_STEP      %d\n\n" % self.s_size)

        f.write("#define KERNEL_NAME %s\n\n" % self.kname)

        f.write("#include \"idxn/int8/const_macros.h\"\n\n")

        if self.s_size == 16:
            f.write("#include \"idxn/int8/dmem_i1_macros.h\"\n\n")
            f.write("#include \"idxn/int8/imma8816_i1_macros.h\"\n\n")

            f.write("#define LOAD_dAv1(_regA, _dAv1, _in_id, _in_off)    LOAD_dAv1_SIZE%d(_regA, _dAv1, _in_id, _in_off)\n" % self.dAvn_size)
            f.write("#define LOAD_dBv1(_regB, _dBv1, _dBv1_off)          LOAD_dBv1_SIZE%d(_regB, _dBv1, _dBv1_off)\n\n" % self.dBvn_size)

            f.write("#define MMA_INSTS(_C, _A, _B)                       MMA_INST_1INT_%dx%d(_C, _A, _B)\n\n" % (self.dAvn_size, self.dBvn_size))
        elif self.s_size == 32:
            f.write("#include \"idxn/int8/dmem_i2_macros.h\"\n\n")
            f.write("#include \"idxn/int8/imma8816_i2_macros.h\"\n\n")

            f.write("#define LOAD_dAv2(_regA, _dAv2, _in_id, _in_off)    LOAD_dAv2_SIZE%d(_regA, _dAv2, _in_id, _in_off)\n" % self.dAvn_size)
            f.write("#define LOAD_dBv2(_regB, _dBv2, _dBv2_off)          LOAD_dBv2_SIZE%d(_regB, _dBv2, _dBv2_off)\n\n" % self.dBvn_size)

            f.write("#define MMA_INSTS(_C, _A, _B)                       MMA_INST_2INT_%dx%d(_C, _A, _B)\n\n" % (self.dAvn_size, self.dBvn_size))
        elif self.s_size == 64:
            f.write("#include \"idxn/int8/dmem_i4_macros.h\"\n\n")
            f.write("#include \"idxn/int8/imma8816_i4_macros.h\"\n\n")

            f.write("#define LOAD_dAv4(_regA, _dAv4, _in_id, _in_off)    LOAD_dAv4_SIZE%d(_regA, _dAv4, _in_id, _in_off)\n" % self.dAvn_size)
            f.write("#define LOAD_dBv4(_regB, _dBv4, _dBv4_off)          LOAD_dBv4_SIZE%d(_regB, _dBv4, _dBv4_off)\n\n" % self.dBvn_size)

            f.write("#define MMA_INSTS(_C, _A, _B)                       MMA_INST_4INT_%dx%d(_C, _A, _B)\n\n" % (self.dAvn_size, self.dBvn_size))

        f.write("#include \"idxn/int8/imma8816_output_macros.h\"\n\n")

        f.write("#include \"idxn/int8/main_body.h\"\n\n")

        f.write("#include \"idxn/int8/uni_undefs.h\"\n\n")

class IdxSourceFile:
    def __init__(self, path):
        self.path = path
        self.fname = "idxn_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include  \"idxn/sm75/int8/imma8816/idxn_kernels.h\"\n\n")

        self.f.write("#define ENABLE_FUSE\n\n")

    def AppendKernel(self, fname):
        self.f.write("#include \"idxn/sm75/int8/imma8816/%s\"\n" % fname)

    def Close(self):
        self.f.close()

class IdxHeaderFile:
    def __init__(self, path):
        self.path = path
        self.fname = "idxn_kernels.h"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#ifndef __PPLCUDA_IDXN_SM75_INT8_IMMA8816_KERNELS_H__\n")
        self.f.write("#define __PPLCUDA_IDXN_SM75_INT8_IMMA8816_KERNELS_H__\n")

        self.f.write("\n\n#include \"kernel_type.h\"\n\n")

    def AppendKernel(self, kname):
        self.f.write("__global__ int8_idx_kernel_t %s;\n" % kname)

    def Close(self):
        self.f.write("\n\n#endif\n")
        self.f.close()

class InitFile:
    def __init__(self, path):
        self.path = path
        self.fname = "init_idxn_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include \"conv_common.h\"\n\n")

        self.f.write("#include \"idxn/sm75/int8/imma8816/idxn_kernels.h\"\n\n")

        self.f.write("void InitializeIdxnSM75Int8Imma8816ConvKernelContainer(std::vector<kernel_info_t> & kernel_container)\n{\n")

    def AppendKernel(self, s_size, kname):
        if s_size == 16:
            self.f.write("\tADD_KERNEL(CONV_IDXN_C4, \"%s\", NULL, NULL, &%s);\n" % (kname, kname))
        elif s_size == 32:
            self.f.write("\tADD_KERNEL(CONV_IDXN_C8, \"%s\", NULL, NULL, &%s);\n" % (kname, kname))
        elif s_size == 64:
            self.f.write("\tADD_KERNEL(CONV_IDXN_C64, \"%s\", NULL, NULL, &%s);\n" % (kname, kname))
        else:
            exit(1)

    def Close(self):
        self.f.write("\n}\n")
        self.f.close()

class HashFile:
    def __init__(self, path, hash_path):
        self.path = path
        self.fname = ".hash_file.txt"

        self.current_hash = dict()
        for root, dirs, files in os.walk(hash_path):
            for file in files:
                fname = os.path.join(root, file)
                fhash = hashlib.md5(open(fname, 'rb').read()).hexdigest()
                self.current_hash[fname] = fhash

    def CheckFileExist(self):
        return os.path.isfile(os.path.join(self.path, self.fname))

    def CompareWithPreviousHash(self):
        previous_hash = dict()

        for line in open(os.path.join(self.path, self.fname), "r"):
            fname, fhash = line.split()
            previous_hash[fname] = fhash

        return previous_hash == self.current_hash

    def WriteCurrentHash(self):
        self.f = open(os.path.join(self.path, self.fname), "w")

        for fname, fhash in self.current_hash.items():
            self.f.write("%s\t%s\n" % (fname, fhash))

    def Close(self):
        self.f.close()

def GenAllKernels(parent_path, kernel_cut=False):
    idx_header_file = IdxHeaderFile(parent_path)
    idx_source_file = IdxSourceFile(parent_path)

    init_file = InitFile(parent_path)

    path = parent_path + '/kernels'

    if not os.path.exists(path):
        os.makedirs(path)

    s_size_l = [16, 32, 64]
    k_num_l = [1, 2]
    warp_y_l = [8, 16, 32, 64, 128]
    warp_x_l = [8, 16, 32, 64]
    cta_y_num_l = [1, 2, 4]
    cta_x_num_l = [1, 2, 4]
    
    if kernel_cut:
        s_size_l = [16, 32, 64]
        k_num_l = [1, 2]
        warp_y_l = [8, 16, 32, 64]
        warp_x_l = [8, 16, 32, 64]
        cta_y_num_l = [1, 2, 4]
        cta_x_num_l = [1, 2, 4]

    for s_size in s_size_l:
        for k_num in k_num_l:
            for warp_y in warp_y_l:
                for warp_x in warp_x_l:
                    for cta_y_num in cta_y_num_l:
                        for cta_x_num in cta_x_num_l:
                            if cta_y_num == 4 and cta_x_num == 4:
                                continue
                            if warp_y == 128 and warp_x == 64:
                                continue

                            kernel = KernelInfo(parent_path, s_size, k_num, cta_y_num, cta_x_num, warp_y, warp_x)

                            if kernel.IsKernelFeasible():
                                kernel.GenKernel()

                                idx_header_file.AppendKernel(kernel.kname)
                                idx_source_file.AppendKernel(kernel.fname)

                                init_file.AppendKernel(s_size, kernel.kname)

    idx_header_file.Close()
    idx_source_file.Close()

    init_file.Close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    kernel_cut = True if sys.argv[2] == "ON" else False

    if not os.path.exists(path):
        os.makedirs(path)

    hash_file = HashFile(path, os.path.dirname(os.path.abspath(__file__)))

    if not hash_file.CheckFileExist() or not hash_file.CompareWithPreviousHash():

        GenAllKernels(path, kernel_cut)

        hash_file.WriteCurrentHash()

        hash_file.Close()
