# ResNet-152 + ABFT 估算报告

## 设定
- 输入：`[1, 3, 224, 224]`
- ABFT（新模式）：A行和 + B列和 + 点乘 + 输出矩阵全元素和校验；不做额外矩阵乘
- 访存：仅权重访存 `params * 2`（BF16）

## 基线统计（按模块）

| 模块 | 矩阵乘 FLOPs | 向量加法 | 向量乘法 | 向量非线性 | 向量总计 | 参数量 | 权重访存(bytes) |
|------|-------------|----------|----------|------------|----------|--------|----------------|
| stem | 2.36e+08 | 1.606e+06 | 8.028e+05 | 3.412e+06 | 5.82e+06 | 9,536 | 1.907e+04 |
| layer1 | 1.336e+09 | 1.124e+07 | 4.415e+06 | 8.028e+06 | 2.368e+07 | 215,808 | 4.316e+05 |
| layer2 | 3.648e+09 | 1.365e+07 | 5.218e+06 | 1.004e+07 | 2.89e+07 | 2,339,840 | 4.68e+06 |
| layer3 | 1.588e+10 | 2.93e+07 | 1.104e+07 | 2.188e+07 | 6.222e+07 | 40,613,888 | 8.123e+07 |
| layer4 | 1.464e+09 | 1.405e+06 | 5.519e+05 | 1.004e+06 | 2.96e+06 | 14,964,736 | 2.993e+07 |
| fc_head | 4.096e+06 | 1.004e+05 | 0 | 0 | 1.004e+05 | 2,049,000 | 4.098e+06 |
| **总和** | 2.256e+10 | 5.73e+07 | 2.203e+07 | 4.436e+07 | 1.237e+08 | 60,192,808 | 1.204e+08 |

## ABFT 后开销与占比（按模块）

| 模块 | 矩阵乘 FLOPs(ABFT) | Δ矩阵乘 | 新增加法 | 新增乘法 | 新增非线性 | 新增向量总计 | 新增总计/基线向量% |
|------|---------------------|---------|----------|----------|------------|--------------|--------------------|
| stem | 2.36e+08 | 0 | 2.644e+06 | 147 | 0 | 2.644e+06 | 45.42% |
| layer1 | 1.336e+09 | 0 | 1.263e+07 | 2560 | 0 | 1.263e+07 | 53.32% |
| layer2 | 3.648e+09 | 0 | 1.877e+07 | 1.434e+04 | 0 | 1.879e+07 | 65.00% |
| layer3 | 1.588e+10 | 0 | 7.688e+07 | 1.29e+05 | 0 | 7.701e+07 | 123.77% |
| layer4 | 1.464e+09 | 0 | 1.656e+07 | 2.15e+04 | 0 | 1.658e+07 | 560.03% |
| fc_head | 4.096e+06 | 0 | 2.052e+06 | 2048 | 0 | 2.054e+06 | 2046.94% |
| **总和** | 2.256e+10 | 0 | 1.295e+08 | 1.696e+05 | 0 | 1.297e+08 | 104.87% |

## Conv->GEMM 维度表（Top 40 by FLOPs）

| module | op | M | K | N | count | FLOPs | conv来源 |
|--------|----|---|---|---|-------|-------|----------|
| stem | stem_conv1 | 12544 | 147 | 64 | 1 | 2.36e+08 | conv7x7 Cin=3 Cout=64 HoutxWout=112x112 stride=2 groups=1 |
| layer1 | layer1_block0_conv2 | 3136 | 576 | 64 | 1 | 2.312e+08 | conv3x3 Cin=64 Cout=64 HoutxWout=56x56 stride=1 groups=1 |
| layer1 | layer1_block1_conv2 | 3136 | 576 | 64 | 1 | 2.312e+08 | conv3x3 Cin=64 Cout=64 HoutxWout=56x56 stride=1 groups=1 |
| layer1 | layer1_block2_conv2 | 3136 | 576 | 64 | 1 | 2.312e+08 | conv3x3 Cin=64 Cout=64 HoutxWout=56x56 stride=1 groups=1 |
| layer2 | layer2_block0_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer2 | layer2_block1_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer2 | layer2_block2_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer2 | layer2_block3_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer2 | layer2_block4_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer2 | layer2_block5_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer2 | layer2_block6_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer2 | layer2_block7_conv2 | 784 | 1152 | 128 | 1 | 2.312e+08 | conv3x3 Cin=128 Cout=128 HoutxWout=28x28 stride=1 groups=1 |
| layer3 | layer3_block0_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block1_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block2_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block3_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block4_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block5_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block6_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block7_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block8_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block9_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block10_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block11_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block12_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block13_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block14_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block15_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block16_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block17_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block18_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block19_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block20_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block21_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block22_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block23_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block24_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block25_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block26_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
| layer3 | layer3_block27_conv2 | 196 | 2304 | 256 | 1 | 2.312e+08 | conv3x3 Cin=256 Cout=256 HoutxWout=14x14 stride=1 groups=1 |
