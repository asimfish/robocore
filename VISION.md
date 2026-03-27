# VISION.md — RoboCore 宏观战略指导

## 生成时间: 2026-03-27 00:00

## 1. 终极目标（一句话）
构建一个 **通用、高效、易扩展** 的机器人学习祖传代码库（RoboCore），统一支持主流 benchmark、算法和数据格式，成为社区标准开源工具。

## 2. 核心价值主张
- **解决什么问题？** 机器人学习领域碎片化严重——每个 benchmark 一套代码、每个算法一个仓库、每种数据格式一套 pipeline，研究者花大量时间在工程对接而非算法创新上。
- **为什么现有方案不够？** LeRobot 偏硬件控制和数据集托管；RoboMimic 只覆盖 imitation learning 子集；ManiSkill 是仿真平台而非算法库；没有一个库能同时覆盖 DP/DP3/ACT/FlowPolicy/OpenVLA/Pi0 + 多 benchmark + 多数据格式。
- **我们的独特优势？** 统一抽象层 + 插件化架构 + 零样板代码切换 benchmark/算法/数据格式。

## 3. 成功标准（可量化）
- [ ] 支持 benchmark 数量: 0 → 4+ (RoboMimic, LIBERO, ManiSkill3, RoboTwin)
- [ ] 支持算法数量: 0 → 6+ (DP, DP3, ACT, FlowPolicy, OpenVLA, Pi0)
- [ ] 支持数据格式: 0 → 4+ (LeRobot, RLDS, HDF5, Zarr)
- [ ] 核心模块测试覆盖率: 0% → 80%+
- [ ] 文档完整度: 0% → 90%+ (API doc + quickstart + tutorials)
- [ ] GitHub Stars 目标: 0 → 500+ (发布后 3 个月)

## 4. 阶段路线图（按优先级）
| 阶段 | 目标 | 预计轮次 | 关键交付物 |
|---|---|---|-----|
| 阶段 1: 架构设计 | 确定核心抽象、模块边界、接口协议 | 1-5 | ARCHITECTURE.md, ADRs, API Contract |
| 阶段 2: 核心骨架 | data/policy/env/trainer 四大模块骨架 | 6-30 | 可运行的最小框架 |
| 阶段 3: 算法实现 | 6 个经典算法 + 2 个 benchmark | 31-80 | DP/ACT/FlowPolicy + RoboMimic/LIBERO |
| 阶段 4: 全面覆盖 | 剩余算法 + benchmark + 数据格式 | 81-140 | 全部 6 算法 + 4 benchmark + 4 数据格式 |
| 阶段 5: 打磨发布 | 文档、测试、CI/CD、开源准备 | 141-200 | PyPI 包 + GitHub Release + 文档站 |

## 5. 技术约束和原则
- **架构原则**: 插件化 > 继承；组合 > 耦合；配置驱动 > 硬编码
- **代码风格**: PEP 8, type hints 全覆盖, Google docstring
- **性能底线**: 数据加载不成为训练瓶颈（prefetch + multiprocess）
- **安全红线**: 不引入不必要的依赖；核心模块零外部依赖（仅 PyTorch + numpy）
- **兼容性**: Python 3.10+, PyTorch 2.0+, CUDA 11.8+

## 6. 风险和应对
| 风险 | 应对 |
|---|---|
| 抽象层过度设计导致难用 | 每个抽象必须有 3 行 quickstart 验证 |
| benchmark 接口变化 | adapter 模式隔离，版本锁定 |
| 算法实现正确性 | 对标原始仓库复现指标 |
| 依赖冲突 | 核心零依赖，benchmark/算法各自声明 extras |

## 7. 当迷失方向时
1. 修复已知 bug（CRITICAL > HIGH > MEDIUM）
2. 补充测试（覆盖率低的模块优先）
3. 性能优化（数据加载 pipeline 优先）
4. 代码重构（最复杂/耦合最高的模块）
5. 文档完善（无文档的公共 API）
6. 新增算法/benchmark 支持
