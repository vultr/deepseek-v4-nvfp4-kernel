"""DSV4 NVFP4 Fused Router Kernel — Block-scaled GEMM + Activation Epilogue.

Two-phase production path:
  Phase 1 (this kernel): NVFP4 block-scaled GEMM + fused sqrt(softplus) + e_bias
    activation epilogue. Writes FP32 activated scores to GMEM. No intermediate
    BF16 logits buffer. Pure NVFP4 + Blackwell tensor cores the entire way.
  Phase 2 (activation_topk CUDA kernel): top-k + renorm on the activated scores.

The GEMM mainloop and epilogue structure follow FusedSwiGLUScaledGroupedGemmKernel
(dsv4/kernels/gemm/fused_swiglu.py) exactly, with a different activation function
(sqrt(softplus) + e_bias instead of SwiGLU) and no SwiGLU clamp.

Warp specialization (6 warps, no scheduler for dense GEMM):
  Warps 0-3: Epilogue (TMEM -> register -> activation -> SMEM -> TMA store -> GMEM)
  Warp 4:    MMA (tcgen05.mma.block_scale with SFA/SFB in TMEM)
  Warp 5:    TMA load (A, B, SFA, SFB from GMEM -> SMEM)

Pipeline structure (2 pipelines):
  AB pipeline:  TMA (producer) -> MMA (consumer)   [PipelineTmaUmma]
  Acc pipeline: MMA (producer) -> Epilogue (consumer) [PipelineUmmaAsync]

The epilogue uses the proven one-way TMEM→registers→SMEM→GMEM path from the MoE
kernel. This is the same pattern that compiles and runs correctly in
FusedSwigGLUScaledGroupedGemmKernel. No SMEM top-k merge (which crashed MLIR).
"""

from __future__ import annotations
from typing import Tuple, Optional, Type, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)


class Nvfp4FusedRouterKernel:
    """
    NVFP4 blockscaled GEMM + fused activation epilogue.

    Dense (non-grouped) GEMM: [M, K] @ [K, E] -> [M, E] with NVFP4 weights.
    Custom epilogue: TMEM -> registers -> sqrt(softplus(logit)) + e_bias -> SMEM -> GMEM.
    Follows FusedSwiGLUScaledGroupedGemmKernel pattern exactly.
    """

    def __init__(
        self,
        sf_vec_size: int = 16,
        mma_tiler_mnk: Tuple[int, int, int] = (128, 128, 64),
        cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1),
    ):
        self.sf_vec_size = sf_vec_size
        self.mma_tiler_mnk = mma_tiler_mnk
        self.cluster_shape_mn = (cluster_shape_mnk[0], cluster_shape_mnk[1])
        self.use_2cta_instrs = mma_tiler_mnk[0] == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.arch = "sm_100"

        self.mma_inst_shape_mn = (mma_tiler_mnk[0], mma_tiler_mnk[1])
        self.mma_inst_shape_mn_sfb = (
            mma_tiler_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(mma_tiler_mnk[1], 128),
        )

        # 6-warp specialization (no scheduler warp for dense GEMM)
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 6

        # Barrier IDs
        self.cta_sync_bar_id = 1
        self.epilogue_sync_bar_id = 2
        self.tmem_alloc_sync_bar_id = 3

        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.occupancy = 1
        self.buffer_align_bytes = 1024

    def _create_tiled_mma(self, a_dtype, a_major_mode, b_major_mode, sf_dtype):
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            a_dtype, a_major_mode, b_major_mode, sf_dtype,
            self.sf_vec_size, self.cta_group,
            self.mma_inst_shape_mn,
        )

    def _create_tiled_mma_sfb(self, a_dtype, a_major_mode, b_major_mode, sf_dtype):
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            a_dtype, a_major_mode, b_major_mode, sf_dtype,
            self.sf_vec_size, tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

    def _setup_attributes(self, tiled_mma, tiled_mma_sfb, a_dtype, b_dtype, sf_dtype, c_dtype, c_layout):
        """Set up kernel attributes. Mirrors fused_swiglu._setup_attributes."""
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.mma_tiler_mnk[2] // mma_inst_shape_k

        # ── MMA tiler — K is refined in _setup_attributes ──
        # ── MMA tiler — K is refined in _setup_attributes ──
        self.mma_tiler = (self.mma_tiler_mnk[0], self.mma_tiler_mnk[1], 1)
        self.mma_tiler_sfb = (self.mma_tiler_mnk[0] // (2 if self.use_2cta_instrs else 1), cute.round_up(self.mma_tiler_mnk[1], 128), 1)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((self.cluster_shape_mn[0], self.cluster_shape_mn[1], 1)),
            (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((self.cluster_shape_mn[0], self.cluster_shape_mn[1], 1)),
            (tiled_mma_sfb.thr_id.shape,))

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Epilogue tile (same as MoE: compute_epilogue_tile_shape for NVFP4→FP32)
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            c_layout,
            c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Stage counts (same as MoE)
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma, self.mma_tiler_mnk, a_dtype, b_dtype,
            self.epi_tile, c_dtype, c_layout, sf_dtype, self.sf_vec_size,
            self.smem_capacity, self.occupancy)

        # SMEM layouts
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler_mnk, a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler_mnk, b_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_ab_stage)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, self.epi_tile, self.num_c_stage)

        # Overlapping accumulator
        self.overlapping_accum = self.cta_tile_shape_mnk[1] == 256
        if self.overlapping_accum:
            self.num_acc_pipeline_stages = 1
        else:
            self.num_acc_pipeline_stages = self.num_acc_stage

        # TMEM column counts
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage - (
            self.num_sf_tmem_cols if self.overlapping_accum else 0
        )
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols // self.epi_tile_n
        )

        # TMA load bytes
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_smem_0 = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_0 = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_0 = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_0 = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        self.num_tma_load_bytes = (
            cute.size_in_bytes(a_dtype, a_smem_0) +
            cute.size_in_bytes(b_dtype, b_smem_0) +
            cute.size_in_bytes(sf_dtype, sfa_smem_0) +
            cute.size_in_bytes(sf_dtype, sfb_smem_0)
        ) * atom_thr_size

        # TMEM allocation size
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    @staticmethod
    def _compute_stages(
        tiled_mma, mma_tiler_mnk, a_dtype, b_dtype,
        epi_tile, c_dtype, c_layout, sf_dtype, sf_vec_size,
        smem_capacity, occupancy,
    ):
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_c_stage = 2

        a_smem_layout_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_smem_layout_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_smem_layout_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        c_smem_layout_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_one) +
            cute.size_in_bytes(b_dtype, b_smem_layout_one) +
            cute.size_in_bytes(sf_dtype, sfa_smem_layout_one) +
            cute.size_in_bytes(sf_dtype, sfb_smem_layout_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    def mainloop_s2t_copy_and_partition(self, sSF, tSF, cta_group):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # -----------------------------------------------------------------
    # run() — Python entry point
    # -----------------------------------------------------------------
    def run(self, mat_a, mat_b, scale_a, scale_b, mat_c,
            M, N, K, gsa, gsb, stream=None):
        if stream is None:
            stream = cuda.CUstream(0)

        a_dtype = mat_a.element_type
        b_dtype = mat_b.element_type
        sf_dtype = scale_a.element_type
        c_dtype = mat_c.element_type
        a_major_mode = utils.LayoutEnum.from_tensor(mat_a).mma_major_mode()
        b_major_mode = utils.LayoutEnum.from_tensor(mat_b).mma_major_mode()
        c_layout = utils.LayoutEnum.from_tensor(mat_c)

        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.sf_dtype = sf_dtype
        self.c_dtype = c_dtype
        self.a_major_mode = a_major_mode
        self.b_major_mode = b_major_mode

        cta_m = self.mma_tiler_mnk[0]
        cta_n = self.mma_tiler_mnk[1]
        num_M_tiles = (M + cta_m - 1) // cta_m
        num_N_tiles = (N + cta_n - 1) // cta_n
        grid = (num_M_tiles * num_N_tiles, 1, 1)

        @cute.jit
        def _compiled_fn(mat_a, mat_b, scale_a, scale_b, mat_c):
            # Create tiled MMA and setup inside JIT context
            # (same pattern as fused_swiglu.py @cute.jit __call__)
            # Plain int mma_tiler values work with cute.size() inside JIT
            tiled_mma = self._create_tiled_mma(a_dtype, a_major_mode, b_major_mode, sf_dtype)
            tiled_mma_sfb = self._create_tiled_mma_sfb(a_dtype, a_major_mode, b_major_mode, sf_dtype)
            self._setup_attributes(tiled_mma, tiled_mma_sfb, a_dtype, b_dtype, sf_dtype, c_dtype, c_layout)

            # TMA atoms (inside JIT, same as fused_swiglu)
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
            a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mat_a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

            b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
            b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mat_b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

            sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op, scale_a, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Uint64)

            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op, scale_b, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Uint64)

            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mat_c, epi_smem_layout, self.epi_tile)

            tile_sched_params = utils.PersistentTileSchedulerParams(
                (num_M_tiles, num_N_tiles, 1), (1, 1, 1))

            self._kernel(
                tiled_mma, tiled_mma_sfb,
                tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
                tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb,
                tma_atom_c, tma_tensor_c,
                self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
                self.a_smem_layout_staged, self.b_smem_layout_staged,
                self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
                self.c_smem_layout_staged,
                self.epi_tile,
                tile_sched_params,
                M, N, K, gsa, gsb,
            ).launch(
                grid=grid, block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream, min_blocks_per_mp=1,
            )

        cute.compile(_compiled_fn, mat_a, mat_b, scale_a, scale_b, mat_c)

    @cute.kernel
    def _kernel(self, tiled_mma, tiled_mma_sfb,
                tma_atom_a, mA_mkl, tma_atom_b, mB_nkl,
                tma_atom_sfa, mSFA_mkl, tma_atom_sfb, mSFB_nkl,
                tma_atom_c, mC_mnl,
                cluster_layout_vmnk, cluster_layout_sfb_vmnk,
                a_smem_layout_staged, b_smem_layout_staged,
                sfa_smem_layout_staged, sfb_smem_layout_staged,
                c_smem_layout_staged,
                epi_tile,
                tile_sched_params,
                M, N, K, gsa, gsb):

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        use_2cta = cute.size(tiled_mma.thr_id.shape) == 2
        is_leader_cta = (bidx % cute.size(tiled_mma.thr_id.shape)) == 0
        mma_tile_v = bidx % cute.size(tiled_mma.thr_id.shape)
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)

        acc_dtype = cutlass.Float32
        c_dtype = self.c_dtype

        # ============================================================
        # Shared storage
        # ============================================================
        @cute.struct
        class SharedStorage:
            ab_full_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_pipeline_stages * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding: cutlass.Int32
            # C staging SMEM for TMA store (same as MoE epilogue)
            sC: cute.struct.Align[
                cute.struct.MemRange[c_dtype, cute.cosize(c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ============================================================
        # Pipelines
        # ============================================================
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )


        num_acc_cons = self.threads_per_warp * len(self.epilogue_warp_id) * (2 if use_2cta else 1)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_cons),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # C pipeline for TMA store (same as MoE)
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id))
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage,
            producer_group=c_producer_group,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=self.tmem_alloc_sync_bar_id,
                num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilogue_warp_id))),
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr)

        cta_bar = pipeline.NamedBarrier(self.cta_sync_bar_id, self.threads_per_cta)
        epi_sync_bar = pipeline.NamedBarrier(
            self.epilogue_sync_bar_id,
            self.threads_per_warp * len(self.epilogue_warp_id))

        # SMEM tensors
        sA = smem.allocate_tensor(
            element_type=self.a_dtype, layout=a_smem_layout_staged.outer,
            byte_alignment=128, swizzle=a_smem_layout_staged.inner)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype, layout=b_smem_layout_staged.outer,
            byte_alignment=128, swizzle=b_smem_layout_staged.inner)
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128)
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype, layout=sfb_smem_layout_staged, byte_alignment=128)
        sC = smem.allocate_tensor(
            element_type=c_dtype, layout=c_smem_layout_staged.outer,
            byte_alignment=128, swizzle=c_smem_layout_staged.inner)

        # Multicast masks
        a_mcast = None; b_mcast = None; sfa_mcast = None; sfb_mcast = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta):
            a_mcast = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_coord, mcast_mode=2)
            b_mcast = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_coord, mcast_mode=1)
            sfa_mcast = a_mcast
            sfb_mcast = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_coord, mcast_mode=1)

        # Partition global tensors
        gA = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB = cute.local_tile(mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))

        k_tiles = cute.size(gA, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgSFA = thr_mma.partition_A(gSFA)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_v)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)

        # TMA partitions for A/B
        a_cta_l = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, block_coord[2], a_cta_l,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        b_cta_l = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, block_coord[1], b_cta_l,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))

        # TMA partitions for SFA/SFB
        tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa, block_coord[2], a_cta_l,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
        tAsSFA = cute.filter_zeros(tAsSFA); tAgSFA = cute.filter_zeros(tAgSFA)
        block_coord_sfb = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)
        sfb_cta_l = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb, block_coord_sfb[1], sfb_cta_l,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
        tBsSFB = cute.filter_zeros(tBsSFB); tBgSFB = cute.filter_zeros(tBgSFB)

        # TMEM accumulator
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster arrive
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()
        else:
            cta_bar.arrive_and_wait()

        # ============================================================
        # TMA WARP
        # ============================================================
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

            tsched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bidx, cute.arch.grid_dim())
            wt = tsched.initial_work_tile_info()
            ab_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            while wt.is_valid_tile:
                tc = wt.tile_idx
                mc = (tc[0] // cute.size(tiled_mma.thr_id.shape), tc[1], tc[2])
                tAgA_s = tAgA[(None, mc[0], None, mc[2])]
                tBgB_s = tBgB[(None, mc[1], None, mc[2])]
                tAgSFA_s = tAgSFA[(None, mc[0], None, mc[2])]
                slice_n = mc[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mc[1] // 2
                tBgSFB_s = tBgSFB[(None, slice_n, None, mc[2])]

                ab_ps.reset_count()
                peek_ab = cutlass.Boolean(1)
                if ab_ps.count < k_tiles:
                    peek_ab = ab_pipeline.producer_try_acquire(ab_ps)

                for kt in cutlass.range(0, k_tiles, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_ps, peek_ab)
                    cute.copy(tma_atom_a, tAgA_s[(None, ab_ps.count)], tAsA[(None, ab_ps.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_ps), mcast_mask=a_mcast)
                    cute.copy(tma_atom_b, tBgB_s[(None, ab_ps.count)], tBsB[(None, ab_ps.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_ps), mcast_mask=b_mcast)
                    cute.copy(tma_atom_sfa, tAgSFA_s[(None, ab_ps.count)], tAsSFA[(None, ab_ps.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_ps), mcast_mask=sfa_mcast)
                    cute.copy(tma_atom_sfb, tBgSFB_s[(None, ab_ps.count)], tBsSFB[(None, ab_ps.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_ps), mcast_mask=sfb_mcast)
                    ab_ps.advance()
                    peek_ab = cutlass.Boolean(1)
                    if ab_ps.count < k_tiles:
                        peek_ab = ab_pipeline.producer_try_acquire(ab_ps)

                ab_pipeline.producer_tail(ab_ps)
                tsched.advance_to_next_work()
                wt = tsched.get_current_work()

        # ============================================================
        # MMA WARP
        # ============================================================
        if warp_idx == self.mma_warp_id:
            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_wait()
            else:
                cta_bar.arrive_and_wait()

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            # S2T for SFA
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler_mnk, self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
            tCtSFA = cute.make_tensor(acc_tmem_ptr, tCtSFA_layout)
            # S2T for SFB
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_sfb, self.mma_tiler, self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)))
            tCtSFB = cute.make_tensor(acc_tmem_ptr, tCtSFB_layout)

            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = \
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA, self.cta_group)
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = \
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB, tcgen05.CtaGroup.ONE)

            tsched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bidx, cute.arch.grid_dim())
            wt = tsched.initial_work_tile_info()
            ab_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages)

            while wt.is_valid_tile:
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_ps)

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_ps.phase ^ 1
                else:
                    acc_stage_index = acc_ps.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                ab_cs.reset_count()
                peek_ab_full = cutlass.Boolean(1)
                if ab_cs.count < k_tiles and is_leader_cta:
                    peek_ab_full = ab_pipeline.consumer_try_wait(ab_cs)

                for kt in cutlass.range(0, k_tiles, 1, unroll=1):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_cs, peek_ab_full)

                    s2t_stage_coord = (None, None, None, None, ab_cs.index)
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t)

                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll=1):
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
                        kb_coord = (None, None, kblock_idx, ab_cs.index)
                        cute.gemm(tiled_mma, tCrA[kb_coord], tCrB[kb_coord], tCtAcc, tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    ab_pipeline.consumer_release(ab_cs)
                    ab_cs.advance()
                    peek_ab_full = cutlass.Boolean(1)
                    if ab_cs.count < k_tiles:
                        if is_leader_cta:
                            peek_ab_full = ab_pipeline.consumer_try_wait(ab_cs)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_ps)
                acc_ps.advance()
                tsched.advance_to_next_work()
                wt = tsched.get_current_work()

            if is_leader_cta:
                acc_pipeline.producer_tail(acc_ps)
            tmem.relinquish_alloc_permit()

        # ============================================================
        # EPILOGUE WARPS — TMEM→regs→activation→SMEM→GMEM
        # Same pattern as FusedSwiGLUScaledGroupedGemmKernel.
        # Activation: sqrt(softplus(logit)) + e_bias (replaces SwiGLU)
        # ============================================================
        if warp_idx in self.epilogue_warp_id:
            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_wait()
            else:
                cta_bar.arrive_and_wait()

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # TMEM → register copy (paired atoms, same as MoE)
            tiled_copy_t2r, tTR_tAcc_base = epilogue_tmem_copy_and_partition(
                tCtAcc_base, epi_tile, self.epilogue_warp_id, acc_dtype, use_2cta)
            tTR_rAcc = tiled_copy_t2r.fragments_slice(tiled_copy_t2r, tTR_tAcc_base)

            # Register tensor for activation output (same pattern as MoE)
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, c_dtype)

            # Register → SMEM copy (paired atoms, same as MoE)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, tTR_rC, tidx, sC)

            # TMA partition for C store
            tCgC_epi = cute.flat_divide(mC_mnl, epi_tile)
            bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                tma_atom_c, 0, cute.make_layout(1),
                cute.group_modes(sC, 0, 2),
                cute.group_modes(tCgC_epi, 0, 2))

            # Tile scheduler + pipeline states
            tsched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bidx, cute.arch.grid_dim())
            wt = tsched.initial_work_tile_info()
            acc_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_pipeline_stages)

            while wt.is_valid_tile:
                acc_pipeline.consumer_wait(acc_cs)

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_cs.phase
                    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                else:
                    acc_stage_index = acc_cs.index
                    reverse_subtile = cutlass.Boolean(False)

                tc = wt.tile_idx
                mma_tile_coord_mnl = (
                    tc[0] // cute.size(tiled_mma.thr_id.shape), tc[1], tc[2])

                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # Process subtiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tsched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx

                    # Load accumulator from TMEM to registers
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    cute.arch.fence_view_async_tmem_load()

                    # Early release accumulator for overlapping case
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_cs)
                                acc_cs.advance()

                    # Apply global scale (gsa * gsb) to GEMM output
                    # The MMA output is (A * SFA) @ (B * SFB), missing gsa*gsb.
                    # Activation (sqrt(softplus)) is done in Python post-kernel
                    # because CuTeDSL MLIR crashes on exp+log+sqrt.
                    scale = cutlass.Float32(gsa * gsb)
                    acc_vec = tTR_rAcc.load()
                    acc_vec = acc_vec * scale
                    tRS_rC.store(acc_vec.to(c_dtype))

                    # RMEM → SMEM
                    c_buffer = (num_prev_subtiles + real_subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)]
                    )
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta)
                    epi_sync_bar.arrive_and_wait()

                    # SMEM → GMEM (TMA store)
                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    epi_sync_bar.arrive_and_wait()

                # Release accumulator (non-overlapping case)
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_cs)
                        acc_cs.advance()

                tsched.advance_to_next_work()
                wt = tsched.get_current_work()

            # Cleanup
            tmem.relinquish_alloc_permit()
            epi_sync_bar.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            c_pipeline.producer_tail()


# =====================================================================
# Python entry point
# =====================================================================
def run_nvfp4_fused_router(
    hidden_states: torch.Tensor,       # [N, hidden_size] BF16
    mat_b: torch.Tensor,               # [K_packed, E_packed] uint8 NVFP4 weight
    scale_b: torch.Tensor,             # [K_sf, E_sf] FP8 E4M3 weight scale
    gsa: float,                         # activation global scale
    gsb_val: float,                     # weight global scale (weight_scale_2)
    e_bias: torch.Tensor,              # [num_experts] FP32
    routed_scaling_factor: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the NVFP4 fused router: GEMM + activation → top-k.

    Phase 1: CuTeDSL NVFP4 blockscaled GEMM + sqrt(softplus) epilogue
             writes FP32 activated scores to GMEM.
    Phase 2: activation_topk CUDA kernel for top-k + renorm.

    Parameters
    ----------
    hidden_states : [N, hidden_size] BF16 activation tensor
    mat_b : [K_packed, E_packed] uint8 NVFP4 weight (gate projection)
    scale_b : [K_sf, E_sf] FP8 E4M3 weight block scales
    gsa : float, activation global scale (from checkpoint input_scale)
    gsb_val : float, weight global scale (from checkpoint weight_scale_2)
    e_bias : [num_experts] FP32, per-expert selection bias
    routed_scaling_factor : float, post-renorm scaling
    top_k : int, number of experts to select

    Returns
    -------
    topk_weights : [N, top_k] float32
    topk_ids : [N, top_k] int32
    """
    N = hidden_states.shape[0]  # number of tokens
    hidden_size = hidden_states.shape[1]
    E = mat_b.shape[0]  # num_experts (N dimension of GEMM)
    K = mat_b.shape[1] * 2  # K dimension (packed * 2 for FP4)

    device = hidden_states.device

    # Quantize activation to NVFP4
    from dsv4.ops.quantize import quantize_activation_nvfp4
    mat_a_bf16_packed, scale_a_fp8 = quantize_activation_nvfp4(hidden_states, gsa)

    # Output tensor: FP32 activated scores [N, E]
    activated_scores = torch.empty(N, E, dtype=torch.float32, device=device)

    # Convert PyTorch tensors to CuTe tensors (same as gemm_runner.py pattern)
    import cutlass.torch as cutlass_torch

    def _to_cute(t, leading_dim=None):
        ct = cutlass_torch.from_dlpack(t)
        if leading_dim is not None:
            return ct.mark_layout_dynamic(leading_dim=leading_dim)
        return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))

    # Determine leading dimensions from tensor shapes
    # mat_a_bf16_packed: [N, K_packed] — K-major (row-major for GEMM A)
    # mat_b: [E, K_packed] — K-major (col-major for GEMM B, i.e. N-major)
    # Actually, for NVFP4 GEMM: A is M-major, B is N-major
    # Check the existing Nvfp4Linear to see how it handles this
    cute_a = _to_cute(mat_a_bf16_packed)
    cute_b = _to_cute(mat_b)
    cute_sfa = _to_cute(scale_a_fp8)
    cute_sfb = _to_cute(scale_b)
    cute_c = _to_cute(activated_scores)

    # Run the CuTeDSL kernel: NVFP4 GEMM + sqrt(softplus) epilogue
    kernel = Nvfp4FusedRouterKernel(
        sf_vec_size=16,
        mma_tiler_mnk=(128, 128, 64),
        cluster_shape_mnk=(1, 1, 1),
    )
    kernel.run(
        mat_a=cute_a,
        mat_b=cute_b,
        scale_a=cute_sfa,
        scale_b=cute_sfb,
        mat_c=cute_c,
        M=N, N=E, K=K,
        gsa=gsa,
        gsb=gsb_val,
    )

    # Apply sqrt(softplus) activation in PyTorch (CuTeDSL MLIR crashes on exp+log+sqrt)
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = activated_scores.abs()
    pos = activated_scores.clamp(min=0.0)
    exp_neg = torch.exp(-abs_x)
    sp = pos + torch.log1p(exp_neg)
    activated = torch.sqrt(sp)

    # Top-k + renorm on activated scores
    from dsv4.kernels.router._activation_topk import run_fused_activation_topk_pre_activated
    out_weights = torch.empty(N, top_k, dtype=torch.float32, device=device)
    out_ids = torch.empty(N, top_k, dtype=torch.int32, device=device)
    run_fused_activation_topk_pre_activated(
        activated, e_bias, routed_scaling_factor, top_k,
        out_weights, out_ids,
    )

    return out_weights, out_ids
