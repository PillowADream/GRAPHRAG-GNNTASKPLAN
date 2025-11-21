"""Prompt Open-sourced LLM to infer task steps and task invocation path"""
import json
import click
import os
import aiohttp
import asyncio
import time
import sys
import re
sys.path.append("../")
from utils import get_cur_time
from utils.runlog import open_jsonl, log_sample

# ----------------- helpers -----------------
def _clean_code_fences(s: str) -> str:
    """strip ```json ... ``` or ``` ... ``` fences if present"""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    return m.group(1) if m else s

def _extract_json_span(s: str) -> str:
    """return the longest {...} or [...] span inside s"""
    s = s.strip()
    # prefer object
    l, r = s.find("{"), s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return s[l:r+1]
    # fallback to list
    l, r = s.find("["), s.rfind("]")
    if l != -1 and r != -1 and r > l:
        return s[l:r+1]
    return s  # give back original; json.loads will raise

def _try_parse_json(text: str):
    """best-effort parse: remove fences, slice outermost json span, then json.loads"""
    text = _clean_code_fences(text)
    text = text.replace("\_", "_")  # keep original repo behavior
    # don't blanket-remove backslashes/newlines (may be valid in JSON strings)
    candidate = _extract_json_span(text)
    return json.loads(candidate)

def _merge_list_of_dicts(obj):
    """if model returns a list like [{'task_steps':...}, {'task_nodes':...}], merge to one dict"""
    if isinstance(obj, list):
        merged = {}
        for item in obj:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if k in merged and isinstance(merged[k], list) and isinstance(v, list):
                    merged[k].extend(v)
                else:
                    merged[k] = v
        return merged
    return obj

def _linear_links(nodes):
    """build a simple linear chain from ordered nodes if links missing"""
    try:
        names = [n["task"] for n in nodes if isinstance(n, dict) and "task" in n]
        return [{"source": names[i], "target": names[i+1]} for i in range(len(names)-1)]
    except Exception:
        return []
# -------------------------------------------


async def inference_one_case(
    input,
    url,
    temperature,
    top_p,
    tool_string,
    write_file,
    llm,
    demo_string,
    resource_type=False,
    api_key=None,
    # logging meta
    log_fp=None,
    ds=None,
    llm_short=None,
):
    user_request = input["user_request"]

    if resource_type:
        prompt = (
            "\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. "
            'The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], '
            '"task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. '
            "Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}], "
            '"task_links": [{"source": "task name i", "target": "task name j"}]} '
        )
        prompt += (
            "\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. "
            "Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps "
            "should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; "
            "\n4. the tool arguments should be align with the input-type field of # TASK LIST #;"
        )
    else:
        prompt = (
            "\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. "
            'The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: \'xxx\' and xxx: \'xxx\'" ], '
            '"task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", '
            '"value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], '
            '"task_links": [{"source": "task name i", "target": "task name j"}]}'
        )
        prompt += (
            "\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. "
            "Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps "
            "should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, "
            "i.e. the order in which the APIs are invoked;"
        )

    prompt += demo_string
    prompt += (
        "\n\n# USER REQUEST #: {{user_request}}\n"
        "Only output ONE JSON object that strictly follows the schema, without any explanation or markdown fences.\n"
        "# RESULT #:"
    )
    final_prompt = tool_string + prompt.replace("{{user_request}}", user_request)

    payload = {
        "model": f"{llm}",
        "messages": [
            {
                "role": "user",
                "content": final_prompt,
            }
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "frequency_penalty": 0,
        "presence_penalty": 1.05,
        "max_tokens": 2000,
        # 关键：禁用流式并要求 JSON
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    st_time = time.time()

    try:
        returned_content = await get_response(url, payload, resource_type, api_key)
    except Exception as e:
        # 失败也落日志
        if log_fp is not None:
            log_sample(
                fp=log_fp, ds=ds, method="direct", llm=llm_short or llm, seed=0,
                sid=input["id"], success=False, latency_sec=None,
                temperature=temperature, top_p=top_p, alpha=None, graphsearch=None,
                err=str(e)
            )
        print(f"Failed #id {input['id']}: {type(e)} {e}")
        raise e

    # 规范化：如果返回是 list-of-dicts，合并成一个 dict
    returned_content = _merge_list_of_dicts(returned_content)

    # ---- schema repair: 缺字段自动兜底 ----
    for k in ("task_steps", "task_nodes", "task_links"):
        if k not in returned_content or returned_content[k] is None:
            returned_content[k] = []
    if not isinstance(returned_content["task_nodes"], list):
        returned_content["task_nodes"] = []
    if not returned_content["task_links"] and returned_content["task_nodes"]:
        returned_content["task_links"] = _linear_links(returned_content["task_nodes"])
    # -------------------------------------

    lat = round(time.time() - st_time, 4)

    res = {"id": input["id"], "user_request": input["user_request"]}
    res["task_steps"] = returned_content["task_steps"]
    res["task_nodes"] = returned_content["task_nodes"]
    res["task_links"] = returned_content["task_links"]
    res["cost_time"] = lat

    write_file.write(json.dumps(res, ensure_ascii=False) + "\n")
    write_file.flush()

    # 成功日志（tokens_* 暂空，后续可接 usage 或本地估计）
    if log_fp is not None:
        log_sample(
            fp=log_fp, ds=ds, method="direct", llm=llm_short or llm, seed=0,
            sid=input["id"], success=True, latency_sec=lat,
            tokens_prompt=None, tokens_completion=None,
            temperature=temperature, top_p=top_p, alpha=None, graphsearch=None,
            task_steps=res["task_steps"], task_nodes=res["task_nodes"], task_links=res["task_links"], err=None
        )


async def get_response(url, payload, resource_type=False, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=300) as response:
            # 有些实现返回非 JSON 错误体，先尝试 text
            txt = await response.text()
            try:
                resp = json.loads(txt)
            except Exception:
                raise Exception(f"HTTP {response.status}: {txt}")

    if response.status == 429:
        raise Exception(f"Rate Limit Error {resp}")
    if response.status != 200:
        raise Exception(f"{resp}")

    origin_content = resp["choices"][0]["message"]["content"]

    # 第一次：直接严格解析
    try:
        return _try_parse_json(origin_content)
    except Exception:
        pass  # 进入重试

    # 重试：让模型只做“格式化为 JSON”，并继续强制 JSON 返回
    if resource_type:
        fmt_prompt = (
            "Please format the result # RESULT # to a strict JSON object. "
            "Requirements:\n"
            "1) Do not change the semantics of task_steps, task_nodes and task_links;\n"
            "2) Return ONE JSON object only, no code fences, no extra text;\n"
            '3) Schema: {\"task_steps\":[...], \"task_nodes\":[{\"task\":\"...\",\"arguments\":[{\"name\":\"...\",\"value\":\"...\"}]}], '
            '\"task_links\":[{\"source\":\"...\",\"target\":\"...\"}]}\n'
            "# RESULT #:\n{{illegal_result}}"
        )
    else:
        fmt_prompt = (
            "Please format the result # RESULT # to a strict JSON object. "
            "Requirements:\n"
            "1) Do not change the meaning of task_steps, task_nodes, task_links;\n"
            "2) Be careful with brackets, output ONE compact JSON object without any explanation or markdown fences;\n"
            '3) Schema: {\"task_steps\":[\"Step x: ...\"], \"task_nodes\":[{\"task\":\"...\",\"arguments\":[{\"name\":\"...\",\"value\":\"...\"}]}], '
            '\"task_links\":[{\"source\":\"...\",\"target\":\"...\"}]}\n'
            "# RESULT #:\n{{illegal_result}}"
        )

    fmt_payload = dict(payload)  # shallow copy
    fmt_payload["messages"] = [
        {
            "role": "user",
            "content": fmt_prompt.replace("{{illegal_result}}", origin_content),
        }
    ]
    fmt_payload["stream"] = False
    fmt_payload["response_format"] = {"type": "json_object"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=fmt_payload, timeout=180) as response:
            txt = await response.text()
            try:
                resp = json.loads(txt)
            except Exception:
                raise Exception(f"HTTP {response.status}: {txt}")

    if response.status == 429:
        raise Exception(f"Rate Limit Error {resp}")
    if response.status != 200:
        raise Exception(f"{resp}")

    content = resp["choices"][0]["message"]["content"]
    try:
        return _try_parse_json(content)
    except json.JSONDecodeError as e:
        raise Exception(f"JSON Decoding Error {e}")


@click.command()
@click.option("--dataset", default="huggingface", help="The directory of the data")
@click.option("--temperature", type=float, default=0.2)
@click.option("--top_p", type=float, default=0.1)
@click.option("--api_addr", type=str, default="localhost")
@click.option("--api_port", type=int, default=11434)  # 默认指向 Ollama
@click.option("--llm", type=str, default="CodeLlama-13b-Instruct-hf")
@click.option("--use_demos", type=int, default=1)
@click.option("--multiworker", type=int, default=4)
def main(dataset, temperature, top_p, api_addr, api_port, llm, use_demos, multiworker):
    print("= " * 20)
    print("## Starting Time:", get_cur_time(), flush=True)

    # 允许用 OPENAI_BASE_URL 覆盖（例如 Ollama: http://127.0.0.1:11434/v1）
    base_url = os.getenv("OPENAI_BASE_URL", None)
    if base_url:
        url = base_url.rstrip("/") + "/chat/completions"
    else:
        url = f"http://{api_addr}:{api_port}/v1/chat/completions"

    api_key = os.getenv("OPENAI_API_KEY", None)

    llm_short_names = {
        "CodeLlama-13b-Instruct-hf": "CodeLlama-13b",
        "vicuna-13b-v1.5": "vicuna-13b",
        "Mistral-7B-Instruct-v0.2": "mistral-7b",
        "CodeLlama-7b-Instruct-hf": "CodeLlama-7b",
        "Baichuan2-13B-Chat": "Baichuan-13b",
    }
    # 兼容未知模型名（如 ollama 的 "codellama:13b-instruct"）
    llm_short = llm_short_names.get(llm, llm)
    prediction_dir = f"../prediction/{dataset}/{llm_short}"

    infer_step_file = f"{prediction_dir}/direct.json"

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    alignment = json.load(open(f"../data/{dataset}/split_ids.json", "r", encoding="utf-8"))["test_ids"]
    alignment_ids = alignment["chain"]

    has_inferenced = []
    if os.path.exists(infer_step_file):
        with open(infer_step_file, "r", encoding="utf-8") as rf:
            for line in rf:
                try:
                    data = json.loads(line)
                    has_inferenced.append(data["id"])
                except Exception:
                    continue

    with open(f"../data/{dataset}/user_requests.json", "r", encoding="utf-8") as user_request_file:
        inputs = []
        for line in user_request_file:
            item = json.loads(line)
            if item["id"] not in has_inferenced and item["id"] in alignment_ids:
                inputs.append(item)

    write_file = open(infer_step_file, "a", encoding="utf-8")
    print(infer_step_file)

    # Prepare Tool String to prompt LLM
    tool_list = json.load(open(f"../data/{dataset}/tool_desc.json", "r", encoding="utf-8"))["nodes"]
    tool_string = "# TASK LIST #:\n"
    for k, tool in enumerate(tool_list):
        tool_string += json.dumps(tool, ensure_ascii=False) + "\n"

    # Prepare Demo(s) String to prompt LLM
    demo_string = ""
    demos_id = []  # 防 use_demos=0 时未定义
    if use_demos:
        demos_id_list = {
            "huggingface": ["10523150", "14611002", "22067492"],
            "multimedia": ["30934207", "20566230", "19003517"],
            "dailylife": ["27267145", "91005535", "38563456"],
            "tmdb": [1],
            "ultratool": ["691"],
        }

        demos_id = demos_id_list[dataset][:use_demos]

        with open(f"../data/{dataset}/data.json", "r", encoding="utf-8") as demos_rf:
            demos = []
            for line in demos_rf:
                data = json.loads(line)
                if data["id"] in demos_id:
                    demo = {
                        "user_request": data["user_request"],
                        "result": {
                            "task_steps": data["task_steps"],
                            "task_nodes": data["task_nodes"],
                            "task_links": data["task_links"],
                        },
                    }
                    demos.append(demo)

        if len(demos) > 0:
            demo_string += "\nHere are provided examples for your reference.\n"
            for demo in demos:
                demo_string += (
                    f'\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n'
                    f'# RESULT #: {json.dumps(demo["result"], ensure_ascii=False)}'
                )

    # Set up multi-worker
    sem = asyncio.Semaphore(multiworker)

    # 打开统一 JSONL 日志
    log_fp = open_jsonl(f"runs/{dataset}/base/direct.jsonl")

    resp_type = dataset in ["huggingface", "multimedia"]

    async def inference_wrapper(
        input, url, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type
    ):
        async with sem:
            await inference_one_case(
                input, url, temperature, top_p, tool_string, write_file, llm, demo_string,
                resource_type, api_key=api_key,
                log_fp=log_fp, ds=dataset, llm_short=llm_short
            )

    if len(inputs) == 0:
        print("All Completed!")
        log_fp.close()
        return
    else:
        print(f"Detected {len(has_inferenced)} has been inferenced, ")
        print(f"Start inferencing {len(inputs)} tasks ... ")

    loop = asyncio.get_event_loop()

    tasks = []
    for input in inputs:
        if input["id"] not in demos_id:
            tasks.append(
                inference_wrapper(
                    input, url, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type=resp_type
                )
            )
        else:
            print(f"Case {input['id']} in {use_demos}-shot examples and thus Skip")

    done, failed = [], []
    results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    for result in results:
        if isinstance(result, Exception):
            print(result)
            failed.append(result)
        else:
            done.append(result)

    print(f"Completed {len(done)} Failed {len(failed)}")
    loop.close()
    log_fp.close()

    print("\n## Finishing Time:", get_cur_time(), flush=True)
    print("= " * 20)
    print("Done!")


if __name__ == "__main__":
    main()