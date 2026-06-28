# 阶段 02 开始前

阶段 2 将在阶段 1 的 `PHAT + LMS` 双特征前端基础上，补齐 IFAN 主干模型，并先完成工程验证，不直接进入训练：

- 双特征输入分支
- 残差学习模块
- 共享注意力融合模块
- 融合后的回归头
- `PHAT + LMS` 前处理到模型的 forward/backward 验证


相比论文中的双残差注入结构，我们采用的 ConvIco-ReLU-ConvIco-LNormIco + residual + ReLU 残差增强模块牺牲了一部分简单场景下的原始响应保留能力，但显著提升了低 SNR 和强混响场景中的特征稳定性。该设计通过归一化后的残差校正抑制前端 PHAT/LMS 特征中的异常峰值，使 pre-readout MABA 在通道读出前获得更稳定的多通道时序证据，因此在 hard scenes 和 LOCATA task5 上优于论文式残差结构。