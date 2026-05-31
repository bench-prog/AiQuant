# 知识索引

> 领域知识的轻量索引。每条用一句话说清核心逻辑。
> 格式：`- **触发关键词**: 一句话核心逻辑 → `文件路径::类名/函数名`（可选）`

## 技术约定

- **特征工程入口**: 所有特征通过 `build_all_features(df)` 统一构建 → `freqtrade/user_data/strategies/features.py::build_all_features`
- **特征列提取**: 训练/推理使用 `get_feature_columns(df)` 自动剔除基础列 → `freqtrade/user_data/strategies/features.py::get_feature_columns`
- **数据服务注册**: 外部数据通过 `data.service.register()` 注册后用 `query()` 查询 → `data/service.py`
- **默认数据源**: `import data.service_defaults` 自动注册 funding_rate 和 open_interest → `data/service_defaults.py`
- **策略数据路径查找**: 策略通过候选路径列表同时支持本地开发和 Docker 环境 → `strategy_ai_model_v1.py` / `strategy_smallcap_v3_regime.py`
- **模型配置一致性**: `feature_config.json` 存储特征列顺序 + scaler 参数，训练/推理必须严格一致 → `freqtrade/user_data/models/feature_config.json`
- **资金费率特征降级**: `add_funding_rate_features` 在列不存在时原样返回，策略不崩溃 → `features.py::add_funding_rate_features`
- **持仓量特征降级**: `add_open_interest_features` 在列不存在时原样返回，策略不崩溃 → `features.py::add_open_interest_features`
- **训练脚本公共模块**: `research/data_utils.py` 封装 `load_training_data()` 和 `merge_external_data()`，训练脚本复用 → `research/data_utils.py`
- **模型配置命名约定**: 配置文件按 `feature_config_{model_type}.json`（lightgbm/lstm），漂移基线按 `drift_baseline_{model_type}.json` → `freqtrade/user_data/models/`
- **时序数据 DataLoader**: 时序模型训练必须设置 `shuffle=False`，否则破坏时间顺序导致数据泄漏
- **pytest pythonpath 配置**: `pytest.ini` 中设置 `pythonpath = . freqtrade/user_data/strategies data` 让测试直接导入策略模块
- **合成测试数据**: 固定 `np.random.seed(42)`，确保可复现；验证 high >= max(open, close) 避免不合理数据
- **EMA 测试断言**: 短期 EMA 的 std > 长期 EMA（更敏感），而非 mean deviation

## 特征清单

| 类别 | 特征名 | 说明 |
|------|--------|------|
| 趋势 | ema_12/26/50, macd/macd_signal/macd_hist | EMA + MACD |
| 动量 | rsi_14/6, stoch_k/d, cci_20 | RSI + 随机指标 + CCI |
| 波动 | atr_14, bb_lower/middle/upper/width | ATR + 布林带 |
| 成交量 | volume_sma_20, volume_ratio, obv, vwap | 成交量分析 |
| K线 | close_above_ema12/26, body_pct, upper/lower_wick_pct | 价格行为 |
| 滞后 | return_lag_1/2/3/5/10, volume_lag_1/2/3/5/10 | 历史收益率/成交量 |
| 时间 | hour, hour_sin/cos | 小时编码 |
| 收益 | return_6h/24h, volatility_12h | 收益率/波动率 |
| 资金费率 | funding_rate, funding_rate_ema_8, sign, change | 资金费率特征 |
| 持仓量 | open_interest, oi_ema_12/24, oi_change_1h/6h/24h, oi_velocity | OI 特征 |

## 相关文档

- **策略演进跟踪**: `cclt/knowledge/strategy-evolution.md` — 策略总览、演进时间线、特征/模型演进、路线图、变更记录
- **开发流程规范**: `cclt/process.md` — cclt 工作流、变更命名规范、演进跟踪规则、审查检查清单

## 踩坑记录

- **指标函数定义 ≠ 特征列生成**: `adx()` 已定义但 `add_trend_features()` 未调用，导致特征缺失。必须 double check 调用链 → `features.py::build_all_features`
- **特征函数间 EMA 重复计算**: `add_candle_features()` 内联计算 EMA 与 `add_trend_features()` 冗余，应复用已有列 → `features.py::add_candle_features`
- **新增特征破坏模型兼容**: 特征列数变化导致旧 `.pkl` / `.pt` 无法加载，需版本管理或重新训练 → `freqtrade/user_data/models/`
- **向后兼容复用模式**: `df["ema_12"] if "ema_12" in df.columns else ema(...)` 是安全的跨函数列复用模式 → `features.py`
- **特征参数集中管理**: `FEATURE_PARAMS` 字典集中管理所有指标窗口参数，超参优化和消融实验可自动化 → `features.py::FEATURE_PARAMS`
- **FeatureRegistry 按需计算**: 按特征组名计算避免全量开销，支持元数据管理和配置驱动 → `features.py::FeatureRegistry`
- **YAML 配置驱动特征**: `load_feature_config()` 支持 "default"/"minimal" 预设，为 A/B 实验和多策略配置打基础 → `features.py::load_feature_config`
- **大时间框架防泄漏**: 4h/1d 特征合并到 1h 时，时间戳需偏移一个周期长度后再 ffill，避免用未完成蜡烛数据 → `features.py::add_higher_timeframe_features`
- **多币种模型部署模式**: `models/<PAIR>/` 子目录结构，每个 pair 独立模型+配置+基线，便于批量部署和版本管理 → `freqtrade/user_data/models/`
- **Freqtrade 策略状态隔离**: 策略单实例按 pair 顺序调用 `populate_indicators()`，必须用 `_activate_pair()` / `_save_pair()` 配对避免状态交叉污染 → `strategy_ai_model_v1.py::populate_indicators`
- **多币种向后兼容**: 新布局下保留 `_load_legacy_model()` 回退路径，保护已有单币种部署 → `strategy_ai_model_v1.py::_load_legacy_model`
- **多时间框架白名单同步**: `informative_pairs()` 返回的白名单必须和主时间框架一致，否则缺少数据导致特征缺失 → `strategy_ai_model_v1.py::informative_pairs`
- **安全除法模式**: pandas Series 除法前用 `.replace(0, np.nan)` 避免除零产生 inf，修复后需 `fillna()` 处理边界值 → `features.py`
- **空字典防御**: Python 中 `{}` 是 truthy，`params or FEATURE_PARAMS` 不会 fallback，必须用 `is not None` 判断 → `features.py`
- **配置一致性**: `trading_mode` 和 `exchange.ccxt_config.options.defaultType` 必须匹配，否则交易行为异常 → `config_*.json`
- **外部数据窗口限制**: 回测时请求全周期外部数据会导致缓存膨胀，应限制为合理窗口（如 90 天）→ `strategy_ai_model_v1.py::populate_indicators`
- **动态仓位公式**: `final = base × confidence × volatility`，base 用钱包百分比，confidence 线性映射，volatility 用 ATR 反比 → `strategy_ai_model_v1.py::custom_stake_amount`
- **仓位边界保护**: 最小仓位（如 20%）防止手续费占比过高，最大仓位（如 200%）防止过度集中 → `strategy_ai_model_v1.py::POSITION_SIZING_CONFIG`
- **纯函数测试策略**: Freqtrade 策略类依赖运行时，核心计算逻辑提取为纯函数便于单元测试 → `tests/test_position_sizing.py`
- **策略组合模式**: 在单一 IStrategy 中集成多个子策略的信号逻辑，通过加权融合生成最终信号 → `strategy_ensemble_v1.py`
- **信号归一化**: 不同策略的信号范围不同（概率 [0,1] vs 布尔 {0,1}），必须统一映射到相同范围才能加权 → `strategy_ensemble_v1.py::_compute_trend_signal`
- **互补性设计**: AI 抓时机 + Trend 抓方向，两者在不同市场状态下各有优势，组合后平滑收益曲线 → `strategy_ensemble_v1.py::ENSEMBLE_WEIGHTS`
