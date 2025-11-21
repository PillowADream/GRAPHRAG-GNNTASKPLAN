# run_m6.ps1

# 设置模型名称和端口 (请根据实际情况修改)
$LLM = "CodeLlama-13b-Instruct-hf"
$PORT = 8008

# 数据集列表 (建议先跑 tmdb 验证)
$DATASETS = @("tmdb", "dailylife", "huggingface")

# 循环运行所有数据集
foreach ($ds in $DATASETS) {
    Write-Host "==========================================="
    Write-Host "Running Experiments for Dataset: $ds"
    Write-Host "==========================================="

    # 1. 确定 Schema 路径和执行模式
    # TMDB 使用真实 OAS 并开启真实执行；其他使用 tool_desc 并强制模拟执行
    if ($ds -eq "tmdb") {
        $SCHEMA = "data/raw/RestBench/tmdb_oas.json"
        $env:TOOLLLM_REAL_EXEC = "1"  # 开启真实执行 (需确保 TMDB_BEARER_TOKEN 已设置)
    } else {
        $SCHEMA = "data/$ds/tool_desc.json"
        $env:TOOLLLM_REAL_EXEC = "0"  # 强制模拟执行
    }

    # --- Exp A: Baseline (Graph Only) ---
    Write-Host "[Exp A] Baseline: Graph Only"
    python -m trainfree.graphsearch `
        --dataset $ds --llm $LLM --api_port $PORT `
        --mode full --overwrite 1 `
        --use_graphrag 0 `
        --use_toolllm 0 `
        --run_tag base

    # --- Exp B: Full RAG (High Cost Baseline) ---
    Write-Host "[Exp B] Full RAG"
    python -m trainfree.graphsearch `
        --dataset $ds --llm $LLM --api_port $PORT `
        --mode full --overwrite 1 `
        --use_graphrag 1 --rag_triggered 0 --rag_mode auto `
        --use_toolllm 0 `
        --run_tag full_rag

    # --- Exp C: Triggered RAG (M5: Cost-Effectiveness) ---
    # 阈值设为 3
    Write-Host "[Exp C] Triggered RAG (Threshold=3)"
    python -m trainfree.graphsearch `
        --dataset $ds --llm $LLM --api_port $PORT `
        --mode full --overwrite 1 `
        --use_graphrag 1 --rag_triggered 1 --gs_conf_th 3 --rag_mode auto `
        --use_toolllm 0 `
        --run_tag trig_rag

    # --- Exp D: Self-Correction (M4: Execution Quality) ---
    # 基于纯图搜索，叠加 ToolLLM 修正
    Write-Host "[Exp D] Self-Correction (M4)"
    python -m trainfree.graphsearch `
        --dataset $ds --llm $LLM --api_port $PORT `
        --mode full --overwrite 1 `
        --use_graphrag 0 `
        --use_toolllm 1 --tool_schema "$SCHEMA" --confirm_top 3 `
        --run_tag self_correct

    Write-Host "Finished $ds."
}

Write-Host "All experiments completed."