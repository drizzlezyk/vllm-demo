from vllm import LLM

# 设置 tensor_parallel_size 参数为 GPU 数量
llm = LLM(model="facebook/opt-13b", tensor_parallel_size=4)

# 生成输出
output = llm.generate("San Francisco is a")
print(output.outputs[0].text)