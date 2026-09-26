# EvalPlus HumanEval+ — qwen38-flashnext-nvfp4 on this stack (2026-09-25)

Result: pass@1 = 0.945 (HumanEval base) / 0.921 (HumanEval+ extended tests)

## Config
- evalplus.codegen qwen38-flashnext-nvfp4 --dataset humaneval
  --backend openai --base_url http://127.0.0.1:8007/v1 --greedy
- Server: this repo's launcher (GMU 0.94, TP2, NVFP4)
- Thinking DISABLED via chat_template_kwargs={"enable_thinking": false}
- NOTE: required a one-line patch to evalplus's
  evalplus/gen/util/openai_request.py (add extra_body with the kwarg) —
  upstream EvalPlus does not send it, and with thinking on the model
  spends the entire 512-token sampling budget on reasoning_content and
  returns empty completions.
- Evaluation ran in the sandboxed ganler/evalplus:latest Docker container
  (generated code never executed on the host).

samples.jsonl = the 164 greedy solutions evaluated.
