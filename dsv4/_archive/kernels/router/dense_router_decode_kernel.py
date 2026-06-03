"""DSV4 Dense Router Decode Kernel — Blackwell BF16 GEMM + fused router epilogue.

Warp-specialized persistent GEMM:
  Warp 5 (TMA):  Load X [M,K] and W_gate [K,E] tiles GMEM -> SMEM
  Warp 4 (MMA):  X @ W_gate in BF16, FP32 accumulator -> TMEM
  Warps 0-3 (EPI): TMEM -> register, sqrt(softplus), bias, top-k, renorm, GMEM store

The epilogue is a ROW-LEVEL top-k reduction (not per-element like EFC).
The top-k heap accumulates across all subtiles, then merge + renorm + store once per row.

Math (DSV4 S2.1):
  logit = X @ W_gate  (BF16 GEMM, FP32 accumulator)
  act  = sqrt(softplus(logit))  softplus(x) = max(x,0) + log(1+exp(-|x|))
  score = act + e_bias[e]
  ids  = argtopk(score, k=6)
  w    = (act[ids] / sum(act[ids])) * scaling
"""

from __future__ import annotations
from typing import Tuple, Type
import types

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch


class DenseRouterDecodeKernel:

    def __init__(self, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), top_k=6):
        self.acc_dtype = cutlass.Float32
        self.a_dtype = cutlass.BFloat16
        self.b_dtype = cutlass.BFloat16
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.top_k = top_k
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 6  # 4 epi + 1 mma + 1 tma

        self.cta_sync_bar_id = 1
        self.epilogue_sync_bar_id = 2
        self.tmem_alloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.occupancy = 1
        self.buffer_align_bytes = 1024

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.acc_dtype, self.cta_group, self.mma_tiler_mn,
        )

    def _setup_attributes(self):
        self._tiled_mma = self._create_tiled_mma()
        mma_inst_shape_k = cute.size(self._tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        k_tile = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (cutlass.Int32(self.mma_tiler_mn[0]), cutlass.Int32(self.mma_tiler_mn[1]), cutlass.Int32(k_tile))
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self._tiled_mma.thr_id.shape),
            self.mma_tiler[1], self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self._tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs,
            layout_d=utils.LayoutEnum.ROW_MAJOR, elem_ty_d=cutlass.Float32,
            layout_c=None, elem_ty_c=None,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        self.num_ab_stage = 2
        self.num_acc_stage = 1

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self._tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self._tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)

        acc_shape = self._tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = self._tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    def run(self, X, W_gate, e_bias, out_w, out_ids, M, E, K, scaling, top_k, stream=None):
        if stream is None:
            stream = cuda.CUstream(0)

        @cute.jit
        def _compiled_fn(X, W_gate, e_bias, out_w, out_ids):
            # Infer major modes from tensor layouts (same as MoE/grouped GEMM kernels)
            self.a_major_mode = utils.LayoutEnum.from_tensor(X).mma_major_mode()
            self.b_major_mode = utils.LayoutEnum.from_tensor(W_gate).mma_major_mode()
            self._setup_attributes()
            tiled_mma = self._tiled_mma
            atom_thr_size = cute.size(tiled_mma.thr_id.shape)
            a_smem_0 = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            a_copy = cute.size_in_bytes(self.a_dtype, a_smem_0)
            b_smem_0 = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
            b_copy = cute.size_in_bytes(self.b_dtype, b_smem_0)
            self.num_tma_load_bytes = (a_copy + b_copy) * atom_thr_size

            # Inside cute.compile, arguments are already CuTe tensors
            X_cu = X
            W_cu = W_gate
            e_bias_cu = e_bias
            out_w_cu = out_w
            out_ids_cu = out_ids

            a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, X_cu, a_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

            b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, W_cu, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

            num_M_tiles = cute.ceil_div(M, self.cta_tile_shape_mnk[0])
            num_N_tiles = cute.ceil_div(E, self.cta_tile_shape_mnk[1])
            L = 1
            grid = (num_M_tiles * num_N_tiles, 1, 1)

            max_active_clusters = 0
            tile_sched_params = utils.PersistentTileSchedulerParams(
                (cutlass.Int32(num_M_tiles), cutlass.Int32(num_N_tiles), cutlass.Int32(L)),
                (*self.cluster_shape_mn, 1))

            self._kernel(
                tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
                self.cluster_layout_vmnk, self.a_smem_layout_staged,
                self.b_smem_layout_staged, self.epi_tile,
                e_bias_cu, out_w_cu, out_ids_cu, tile_sched_params,
                M, E, K, top_k, scaling,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                     cluster=(*self.cluster_shape_mn, 1), stream=stream, min_blocks_per_mp=1)

        cute.compile(_compiled_fn, X, W_gate, e_bias, out_w, out_ids)

    @cute.kernel
    def _kernel(self, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl,
                cluster_layout_vmnk, a_smem_layout_staged, b_smem_layout_staged,
                epi_tile, e_bias_tensor, out_w_tensor, out_id_tensor,
                tile_sched_params, M, N, K, top_k, routed_scaling_factor):

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        use_2cta = cute.size(tiled_mma.thr_id.shape) == 2
        mma_tile_v = bidx % cute.size(tiled_mma.thr_id.shape)
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)

        # Shared storage
        @cute.struct
        class SharedStorage:
            ab_full_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding: cutlass.Int32
            heap_scores: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4*32*6], 128]
            heap_indices: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4*32*6], 128]
            heap_acts: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4*32*6], 128]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_layout_staged.outer)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_layout_staged.outer)], self.buffer_align_bytes]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Pipelines
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk)

        num_acc_cons = self.threads_per_warp * len(self.epilogue_warp_id) * (2 if use_2cta else 1)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_cons),
            cta_layout_vmnk=cluster_layout_vmnk)

        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=self.tmem_alloc_sync_bar_id,
                num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilogue_warp_id))),
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr)

        cta_bar = pipeline.NamedBarrier(self.cta_sync_bar_id, self.threads_per_cta)
        epi_bar = pipeline.NamedBarrier(self.epilogue_sync_bar_id,
                                         self.threads_per_warp * len(self.epilogue_warp_id))

        # SMEM tensors
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        # Multicast masks
        a_mcast = None; b_mcast = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta):
            a_mcast = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_coord, mcast_mode=2)
            b_mcast = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_coord, mcast_mode=1)

        # Partition globals
        gA = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None,0,None)), (None,None,None))
        gB = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0,None,None)), (None,None,None))
        k_tiles = cute.size(gA, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_v)
        tCgA = thr_mma.partition_A(gA); tCgB = thr_mma.partition_B(gB)

        a_cta_l = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0,0,None,0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, block_coord[2], a_cta_l,
            cute.group_modes(sA,0,3), cute.group_modes(tCgA,0,3))
        b_cta_l = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0,None,0,0)).shape)
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, block_coord[1], b_cta_l,
            cute.group_modes(sB,0,3), cute.group_modes(tCgB,0,3))

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # === TMA WARP ===
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            tsched = utils.StaticPersistentTileScheduler.create(tile_sched_params, bidx, cute.arch.grid_dim())
            wt = tsched.initial_work_tile_info()
            ab_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            while wt.is_valid_tile:
                tc = wt.tile_idx
                mc = (tc[0] // cute.size(tiled_mma.thr_id.shape), tc[1], tc[2])
                tA_s = tAgA[(None, mc[0], None, mc[2])]
                tB_s = tBgB[(None, mc[1], None, mc[2])]
                for kt in cutlass.range(0, k_tiles, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_ps)
                    cute.copy(tma_atom_a, tA_s[(None, ab_ps.count)], tAsA[(None, ab_ps.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_ps), mcast_mask=a_mcast)
                    cute.copy(tma_atom_b, tB_s[(None, ab_ps.count)], tBsB[(None, ab_ps.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_ps), mcast_mask=b_mcast)
                    ab_ps.advance()
                ab_pipeline.producer_tail(ab_ps)
                tsched.advance_to_next_work(); wt = tsched.get_current_work()

        # === MMA WARP ===
        if warp_idx == self.mma_warp_id:
            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_wait()
            else:
                cta_bar.arrive_and_wait()
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc = tCtAcc_base[(None, None, None, 0)]
            tsched = utils.StaticPersistentTileScheduler.create(tile_sched_params, bidx, cute.arch.grid_dim())
            wt = tsched.initial_work_tile_info()
            ab_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            while wt.is_valid_tile:
                cute.clear(tCtAcc)
                for kt in cutlass.range(0, k_tiles, 1, unroll=1):
                    ab_pipeline.consumer_wait(ab_cs)
                    cute.copy(tiled_mma.partition_A(sA[(None,None,None,ab_cs.index)]), tCrA)
                    cute.copy(tiled_mma.partition_B(sB[(None,None,None,ab_cs.index)]), tCrB)
                    cute.gemm(tiled_mma, tCrA, tCrB, tCtAcc)
                    ab_pipeline.consumer_release(ab_cs); ab_cs.advance()
                acc_pipeline.producer_commit(acc_ps); acc_ps.advance()
                tsched.advance_to_next_work(); wt = tsched.get_current_work()
            tmem.relinquish_alloc_permit()

        # === EPILOGUE WARPS ===
        if warp_idx in self.epilogue_warp_id:
            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_wait()
            else:
                cta_bar.arrive_and_wait()
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc0 = tCtAcc_base[(None, None, None, 0)]

            # TMEM->register copy (tcgen05.ld)
            epi_n = self.epi_tile_n
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(epi_n)), self.acc_dtype)
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tCtAcc0)
            sfw_idx = tidx % (self.threads_per_warp * len(self.epilogue_warp_id))
            thr_ld = tiled_tmem_load.get_slice(sfw_idx)
            tS = thr_ld.partition_S(tCtAcc0)
            tD = thr_ld.partition_D(tS)

            # Identity tensor for expert index mapping
            cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], N))
            tCcAcc = tiled_mma.get_slice(mma_tile_v).partition_C(cAcc)
            cFlat = cute.flatten(tCcAcc)
            rFlat = cute.flatten(tD)

            # Per-thread register heap (6 entries, scalars for CuTeDSL)
            hs = [cutlass.Float32(-1e30)] * 6
            hi = [cutlass.Int32(-1)] * 6
            ha = [cutlass.Float32(0.0)] * 6

            tsched = utils.StaticPersistentTileScheduler.create(tile_sched_params, bidx, cute.arch.grid_dim())
            wt = tsched.initial_work_tile_info()
            acc_cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            while wt.is_valid_tile:
                acc_pipeline.consumer_wait(acc_cs)
                # Reset heap
                for i in cutlass.range(6, unroll=1):
                    hs[i] = cutlass.Float32(-1e30)
                    hi[i] = cutlass.Int32(-1)
                    ha[i] = cutlass.Float32(0.0)

                # Process subtiles
                for subtile in cutlass.range(1, unroll=1):
                    cute.copy(tiled_tmem_load, tS, tD)

                    elem_cnt = cute.size(rFlat)
                    for e in cutlass.range(elem_cnt, unroll=4):
                        logit = rFlat[e]
                        coord = cFlat[e]
                        e_idx = coord[1]

                        # sqrt(softplus(logit))
                        abs_x = cute.math.absf(logit)
                        pos = cute.where(logit > cutlass.Float32(0.0), logit, cutlass.Float32(0.0))
                        exp_neg = cute.math.exp(-abs_x)
                        sp = pos + cute.math.log(cutlass.Float32(1.0) + exp_neg)
                        act = cute.math.sqrt(sp)

                        # score = act + bias
                        score = act + e_bias_tensor[e_idx]

                        # Min-heap push: root = hs[0] (smallest)
                        do_push = (score > hs[0]) or (score == hs[0] and e_idx < hi[0])
                        if do_push:
                            # Replace root
                            old_s = hs[0]; old_i = hi[0]; old_a = ha[0]
                            hs[0] = score; hi[0] = e_idx; ha[0] = act
                            # Sift down (k=6, fully unrolled)
                            # Depth 0: children 1,2
                            root = 0
                            _done = cutlass.Bool(False)
                            while root < 3 and not _done:
                                left = 2*root+1; right = 2*root+2
                                smallest = root
                                if left < 6:
                                    if hs[left] < hs[smallest] or (hs[left] == hs[smallest] and hi[left] > hi[smallest]):
                                        smallest = left
                                if right < 6:
                                    if hs[right] < hs[smallest] or (hs[right] == hs[smallest] and hi[right] > hi[smallest]):
                                        smallest = right
                                if smallest == root:
                                    _done = cutlass.Bool(True)
                                if not _done:
                                    ts = hs[root]; ti = hi[root]; ta = ha[root]
                                    hs[root] = hs[smallest]; hi[root] = hi[smallest]; ha[root] = ha[smallest]
                                    hs[smallest] = ts; hi[smallest] = ti; ha[smallest] = ta
                                    root = smallest

                # Write heap to shared memory for merge
                tid = (warp_idx * 32 + tidx)
                base = tid * 6
                for i in cutlass.range(6, unroll=1):
                    storage.heap_scores.data_ptr()[base + i] = hs[i]
                    storage.heap_indices.data_ptr()[base + i] = hi[i]
                    storage.heap_acts.data_ptr()[base + i] = ha[i]

                epi_bar.arrive_and_wait()

                # Thread 0 of warp 0 does the final merge + store
                if warp_idx == 0 and tidx == 0:
                    # Initialize final heap from thread 0
                    fs = list(hs); fi = list(hi); fa = list(ha)
                    # Merge all 128 threads
                    for t in cutlass.range(1, 128, unroll=1):
                        for i in cutlass.range(6, unroll=1):
                            cs = storage.heap_scores.data_ptr()[t*6+i]
                            ci = storage.heap_indices.data_ptr()[t*6+i]
                            ca = storage.heap_acts.data_ptr()[t*6+i]
                            if ci >= 0:
                              if cs > fs[0] or (cs == fs[0] and ci < fi[0]):
                                fs[0] = cs; fi[0] = ci; fa[0] = ca
                                # Sift down
                                r = 0
                                _done2 = cutlass.Bool(False)
                                while r < 3 and not _done2:
                                    l = 2*r+1; ri = 2*r+2; sm = r
                                    if l < 6:
                                        if fs[l] < fs[sm] or (fs[l] == fs[sm] and fi[l] > fi[sm]):
                                            sm = l
                                    if ri < 6:
                                        if fs[ri] < fs[sm] or (fs[ri] == fs[sm] and fi[ri] > fi[sm]):
                                            sm = ri
                                    if sm == r:
                                        _done2 = cutlass.Bool(True)
                                    else:
                                        ts=fs[r]; ti=fi[r]; ta=fa[r]
                                        fs[r]=fs[sm]; fi[r]=fi[sm]; fa[r]=fa[sm]
                                        fs[sm]=ts; fi[sm]=ti; fa[sm]=ta
                                        r = sm

                    # Sort descending (selection sort, k=6)
                    sorted_s = [cutlass.Float32(-1e30)]*6
                    sorted_i = [cutlass.Int32(-1)]*6
                    sorted_a = [cutlass.Float32(0.0)]*6
                    for i in cutlass.range(6, unroll=1):
                        best = 0
                        for j in cutlass.range(1, 6, unroll=1):
                            if fs[j] > fs[best] or (fs[j] == fs[best] and fi[j] < fi[best]):
                                best = j
                        sorted_s[i] = fs[best]; sorted_i[i] = fi[best]; sorted_a[i] = fa[best]
                        fs[best] = cutlass.Float32(-1e30)

                    # Renormalize
                    act_sum = sorted_a[0] + sorted_a[1] + sorted_a[2] + sorted_a[3] + sorted_a[4] + sorted_a[5]
                    inv_sum = cutlass.Float32(1.0) / act_sum
                    sc = cutlass.Float32(routed_scaling_factor)

                    # Get tile coordinates for output indexing
                    tc = wt.tile_idx
                    row_base = tc[0] // cute.size(tiled_mma.thr_id.shape) * self.cta_tile_shape_mnk[0]

                    # Store to GMEM
                    for i in cutlass.range(6, unroll=1):
                        out_w_tensor[row_base + 0, i] = sorted_a[i] * inv_sum * sc
                        out_id_tensor[row_base + 0, i] = sorted_i[i]

                epi_bar.arrive_and_wait()

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_cs)
                acc_cs.advance()
                tsched.advance_to_next_work(); wt = tsched.get_current_work()

            # Cleanup
            tmem.relinquish_alloc_permit()
            epi_bar.arrive_and_wait()
            tmem.free(tmem_ptr)
