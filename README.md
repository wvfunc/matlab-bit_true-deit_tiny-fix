# DeiT‑Tiny MATLAB Bit‑True (INT8) — PipeViT 对齐验证

这个仓库提供 **DeiT‑Tiny 全模型 INT8 定点/bit‑true 推理** 的 MATLAB 实现（含 LayerNorm、Softmax‑LUT、GELU‑LUT、Conv/FC 等），
可与在 CPU+FPGA 异构系统上的 **PipeViT** 部署 **逐层对齐（check pass）**。

