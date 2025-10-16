# launch the offline engine
import asyncio
import requests
import torch

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


model_name = "/fs-computility/llm_fudan/shared/models/Qwen2.5-Math/Qwen2.5-Math-PRM-7B"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

data = {
    "system": "Please reason step by step, and put your final answer within \\boxed{}.",
    "query": "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
    "response": [
    "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
    "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
    "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
    "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30})."
    ]
}

messages = [
    {"role": "system", "content": data['system']},
    {"role": "user", "content": data['query']},
    {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
]
conversation_str = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False
)


url = "http://127.0.0.1:30000"

payload = {
    "model": model_name,
    "input": conversation_str,
    "encoding_format": "float"
}

# response = requests.post(url + "/v1/embeddings", json=payload).json()
response = requests.post(url + "/v1/embeddings", json=payload)

# print("Embedding:", response["data"][0]["embedding"])
print("Embedding:", response)


# input_ids = tokenizer.encode(
#     conversation_str, 
#     return_tensors="pt", 
# )

# step_sep_id = tokenizer.encode("<extra_0>")[0]
# token_masks = (input_ids == step_sep_id)
# step_reward = make_step_rewards(response["data"][0]["embedding"], token_masks)
# print(step_reward)  # [[1.0, 0.1904296875, 0.9765625, 1.0]]